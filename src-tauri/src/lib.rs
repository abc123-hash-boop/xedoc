use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Mutex,
};
use tauri::{Emitter, Manager};

#[derive(Default)]
struct AppState {
    config: Mutex<AgentConfig>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
#[serde(default)]
struct ModelProvider {
    id: String,
    name: String,
    api_base: String,
    model: String,
    #[serde(default)]
    models: Vec<String>,
    api_key: String,
}

#[derive(Clone, Serialize, Deserialize, Default)]
#[serde(default)]
struct McpServerConfig {
    id: String,
    name: String,
    transport: String,
    url: String,
    headers: std::collections::HashMap<String, String>,
    command: String,
    args: Vec<String>,
    env: std::collections::HashMap<String, String>,
    enabled: bool,
}

#[derive(Clone, Serialize, Deserialize, Default)]
#[serde(default)]
struct AgentSkill {
    id: String,
    name: String,
    description: String,
    instructions: String,
    enabled: bool,
}

#[derive(Clone, Serialize, Deserialize, Default)]
#[serde(default)]
struct AgentConfig {
    providers: Vec<ModelProvider>,
    active_provider_id: String,
    system_prompt: String,
    mcp_servers: Vec<McpServerConfig>,
    skills: Vec<AgentSkill>,
    // Legacy fields are retained so existing installations migrate without losing settings.
    api_base: String,
    model: String,
    api_key: String,
}

#[derive(Serialize)]
struct AgentStatus {
    ready: bool,
    platform: String,
    version: &'static str,
}

fn config_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_config_dir()
        .map(|p| p.join("config.json"))
        .map_err(|e| e.to_string())
}
fn normalize_config(mut config: AgentConfig) -> AgentConfig {
    if config.providers.is_empty()
        && (!config.api_key.is_empty() || !config.model.is_empty() || !config.api_base.is_empty())
    {
        config.providers.push(ModelProvider {
            id: "default".into(),
            name: "Default provider".into(),
            api_base: if config.api_base.is_empty() {
                "https://api.openai.com/v1".into()
            } else {
                config.api_base.clone()
            },
            model: config.model.clone(),
            models: if config.model.trim().is_empty() { Vec::new() } else { vec![config.model.clone()] },
            api_key: config.api_key.clone(),
        });
        config.active_provider_id = "default".into();
    }
    for provider in &mut config.providers {
        if provider.models.is_empty() && !provider.model.trim().is_empty() { provider.models.push(provider.model.clone()); }
        if provider.model.trim().is_empty() { provider.model = provider.models.first().cloned().unwrap_or_default(); }
        if !provider.model.is_empty() && !provider.models.contains(&provider.model) { provider.models.insert(0, provider.model.clone()); }
    }
    if config.active_provider_id.is_empty() {
        config.active_provider_id = config
            .providers
            .first()
            .map(|p| p.id.clone())
            .unwrap_or_default();
    }
    config
}
fn load_config(app: &tauri::AppHandle) -> AgentConfig {
    config_path(app)
        .ok()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str::<AgentConfig>(&s).ok())
        .map(normalize_config)
        .unwrap_or_default()
}
fn active_provider(config: &AgentConfig) -> Option<ModelProvider> {
    config
        .providers
        .iter()
        .find(|p| p.id == config.active_provider_id)
        .cloned()
}
#[tauri::command]
async fn fetch_models(provider: ModelProvider) -> Result<Vec<String>, String> {
    let base = provider.api_base.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err("Enter an API base URL first.".into());
    }
    let mut request = reqwest::Client::new().get(format!("{base}/models"));
    if !provider.api_key.trim().is_empty() {
        request = request.bearer_auth(provider.api_key.trim());
    }
    let response = request
        .send()
        .await
        .map_err(|e| format!("Could not reach provider: {e}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| format!("Could not read provider response: {e}"))?;
    let value: Value = serde_json::from_str(&body)
        .map_err(|_| format!("Provider returned an invalid models response (HTTP {status})."))?;
    if !status.is_success() {
        let message = value["error"]["message"]
            .as_str()
            .or_else(|| value["message"].as_str())
            .unwrap_or("Provider rejected the models request.");
        return Err(format!("HTTP {status}: {message}"));
    }
    let entries = value["data"]
        .as_array()
        .or_else(|| value["models"].as_array())
        .ok_or("Unsupported models response: expected a 'data' or 'models' array.")?;
    let mut models: Vec<String> = entries
        .iter()
        .filter_map(|item| item["id"].as_str().or_else(|| item["name"].as_str()))
        .map(str::to_string)
        .filter(|id| !id.trim().is_empty())
        .collect();
    models.sort();
    models.dedup();
    if models.is_empty() {
        return Err("The provider returned no model IDs.".into());
    }
    Ok(models)
}
#[tauri::command]
fn status() -> AgentStatus {
    AgentStatus {
        ready: true,
        platform: std::env::consts::OS.to_string(),
        version: "0.1.0",
    }
}
#[tauri::command]
fn get_config(app: tauri::AppHandle, state: tauri::State<AppState>) -> Result<AgentConfig, String> {
    let mut c = state.config.lock().map_err(|e| e.to_string())?;
    if c.providers.is_empty() {
        *c = load_config(&app);
    }
    Ok(c.clone())
}
#[tauri::command]
fn save_config(
    app: tauri::AppHandle,
    state: tauri::State<AppState>,
    config: AgentConfig,
) -> Result<(), String> {
    let config = normalize_config(config);
    let p = config_path(&app)?;
    fs::create_dir_all(p.parent().ok_or("invalid config path")?).map_err(|e| e.to_string())?;
    fs::write(
        p,
        serde_json::to_vec_pretty(&config).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    *state.config.lock().map_err(|e| e.to_string())? = config;
    Ok(())
}
#[tauri::command]
async fn choose_project(app: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().pick_folder(move |folder| {
        let _ = tx.send(folder.map(|path| path.to_string()));
    });
    rx.await.map_err(|e| format!("Folder picker failed: {e}"))
}
fn within(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let p = PathBuf::from(rel);
    let p = if p.is_absolute() { p } else { root.join(p) };
    if p.exists() {
        p.canonicalize().map_err(|e| e.to_string())
    } else {
        let parent = p.parent().unwrap_or(root);
        parent
            .canonicalize()
            .map(|dir| dir.join(p.file_name().unwrap_or_default()))
            .map_err(|e| e.to_string())
    }
}
fn tool_exec(
    name: &str,
    args: &Value,
    root: &Path,
    app: &tauri::AppHandle,
) -> Result<String, String> {
    match name {
        "list_files" => {
            let dir = within(root, args["path"].as_str().unwrap_or("."))?;
            let mut out = String::new();
            for e in fs::read_dir(dir).map_err(|e| e.to_string())? {
                let e = e.map_err(|e| e.to_string())?;
                out.push_str(&format!(
                    "{}{}\n",
                    if e.file_type().map_err(|e| e.to_string())?.is_dir() {
                        "[DIR] "
                    } else {
                        "[FILE] "
                    },
                    e.file_name().to_string_lossy()
                ));
            }
            Ok(out)
        }
        "read_file" => {
            let p = within(root, args["path"].as_str().ok_or("missing path")?)?;
            fs::read_to_string(p).map_err(|e| e.to_string())
        }
        "write_file" => {
            let p = within(root, args["path"].as_str().ok_or("missing path")?)?;
            use tauri_plugin_dialog::DialogExt;
            let message = if p.exists() {
                format!(
                    "Forge wants to overwrite this file:\n\n{}\n\nApprove this change?",
                    p.display()
                )
            } else {
                format!(
                    "Forge wants to create this file:\n\n{}\n\nApprove this change?",
                    p.display()
                )
            };
            if !app
                .dialog()
                .message(message)
                .buttons(tauri_plugin_dialog::MessageDialogButtons::OkCancel)
                .blocking_show()
            {
                return Ok("File change denied by user.".into());
            }
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::write(p, args["content"].as_str().unwrap_or("")).map_err(|e| e.to_string())?;
            Ok("File written.".into())
        }
        "git_diff" => {
            let o = Command::new("git")
                .args(["diff", "--"])
                .current_dir(root)
                .output()
                .map_err(|e| e.to_string())?;
            Ok(String::from_utf8_lossy(&o.stdout).to_string())
        }
        "run_command" => {
            let cmd = args["command"].as_str().ok_or("missing command")?;
            use tauri_plugin_dialog::DialogExt;
            if !app.dialog().message(format!("Forge wants to run this command in the selected workspace:\n\n{cmd}\n\nApprove command execution?")).buttons(tauri_plugin_dialog::MessageDialogButtons::OkCancel).blocking_show() {
                return Ok("Command denied by user.".into());
            }
            let mut c = if cfg!(windows) {
                let mut c = Command::new("cmd");
                c.args(["/C", cmd]);
                c
            } else {
                let mut c = Command::new("sh");
                c.args(["-c", cmd]);
                c
            };
            let o = c.current_dir(root).output().map_err(|e| e.to_string())?;
            Ok(format!(
                "Exit: {}\n{}\n{}",
                o.status,
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            ))
        }
        _ => Err(format!("Unknown tool: {name}")),
    }
}
#[tauri::command]
fn show_git_diff(project_path: String) -> Result<String, String> {
    let root = PathBuf::from(project_path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let o = Command::new("git")
        .args(["diff", "--"])
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&o.stdout).to_string())
}
#[derive(Clone, Deserialize)]
struct ChatTurn {
    role: String,
    content: String,
}
async fn read_stream_response(
    response: reqwest::Response,
    app: &tauri::AppHandle,
    session_id: &str,
) -> Result<Value, String> {
    let status = response.status();
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut content = String::new();
    let mut calls: Vec<Value> = Vec::new();
    let mut raw_body = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Streaming response failed: {e}"))?;
        let text = String::from_utf8_lossy(&chunk);
        raw_body.push_str(&text);
        buffer.push_str(&text);
        while let Some(pos) = buffer.find("\n") {
            let line = buffer[..pos].trim_end_matches("\r").to_string();
            buffer.drain(..=pos);
            if let Some(data) = line.strip_prefix("data:") {
                let data = data.trim();
                if data == "[DONE]" {
                    continue;
                }
                let Ok(event) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                let delta_object = &event["choices"][0]["delta"];
                let reasoning_delta = delta_object["reasoning_content"].as_str()
                    .or_else(|| delta_object["reasoning"].as_str());
                if let Some(delta) = reasoning_delta {
                    if !delta.is_empty() {
                        let _ = app.emit("agent-reasoning", json!({"session_id":session_id,"content":delta}));
                    }
                }
                if let Some(delta) = event["choices"][0]["delta"]["content"].as_str() {
                    if !delta.is_empty() {
                        content.push_str(delta);
                        let _ = app.emit(
                            "agent-token",
                            json!({"session_id":session_id,"content":delta}),
                        );
                    }
                }
                if let Some(deltas) = event["choices"][0]["delta"]["tool_calls"].as_array() {
                    for item in deltas {
                        let index = item["index"].as_u64().unwrap_or(0) as usize;
                        while calls.len() <= index {
                            calls.push(json!({"id":"","type":"function","function":{"name":"","arguments":""}}));
                        }
                        if let Some(v) = item["id"].as_str() {
                            calls[index]["id"] = json!(v);
                        }
                        if let Some(v) = item["function"]["name"].as_str() {
                            let old = calls[index]["function"]["name"].as_str().unwrap_or("");
                            calls[index]["function"]["name"] = json!(format!("{old}{v}"));
                        }
                        if let Some(v) = item["function"]["arguments"].as_str() {
                            let old = calls[index]["function"]["arguments"].as_str().unwrap_or("");
                            calls[index]["function"]["arguments"] = json!(format!("{old}{v}"));
                        }
                    }
                }
            }
        }
    }
    if !status.is_success() {
        return Err(format!(
            "Provider API error (HTTP {status}): {}",
            raw_body.chars().take(800).collect::<String>()
        ));
    }
    if content.is_empty() && calls.is_empty() {
        if let Ok(body) = serde_json::from_str::<Value>(&raw_body) {
            if let Some(message) = body["choices"][0]["message"].as_object() {
                if let Some(reasoning_text) = message.get("reasoning_content").and_then(Value::as_str)
                    .or_else(|| message.get("reasoning").and_then(Value::as_str)) {
                    if !reasoning_text.is_empty() {
                        let _ = app.emit("agent-reasoning", json!({"session_id":session_id,"content":reasoning_text}));
                    }
                }
                return Ok(Value::Object(message.clone()));
            }
            if body["error"].is_object() {
                return Err(body["error"]["message"]
                    .as_str()
                    .unwrap_or("Provider returned an API error.")
                    .to_string());
            }
        }
    }
    let mut message = json!({"role":"assistant","content":content});
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls);
    }
    Ok(message)
}
fn load_workspace_skills(root: &Path) -> Vec<(String, String, String)> {
    let mut found = Vec::new();
    let skills_dir = root.join(".agents").join("skills");
    let Ok(entries) = fs::read_dir(skills_dir) else { return found; };
    for entry in entries.flatten().take(50) {
        let path = entry.path().join("SKILL.md");
        if !path.is_file() { continue; }
        let Ok(content) = fs::read_to_string(&path) else { continue; };
        if content.len() > 64 * 1024 { continue; }
        let folder_name = entry.file_name().to_string_lossy().to_string();
        let mut name = folder_name.clone();
        let mut description = String::new();
        let mut instructions = content.as_str();
        if let Some(rest) = content.strip_prefix("---") {
            if let Some(end) = rest.find("\n---") {
                let frontmatter = &rest[..end];
                for line in frontmatter.lines() {
                    if let Some(value) = line.strip_prefix("name:") { name = value.trim().trim_matches('"').to_string(); }
                    if let Some(value) = line.strip_prefix("description:") { description = value.trim().trim_matches('"').to_string(); }
                }
                instructions = rest[end + 4..].trim_start();
            }
        }
        found.push((name, description, instructions.to_string()));
    }
    found
}

#[tauri::command]
async fn run_task(
    app: tauri::AppHandle,
    prompt: String,
    project_path: Option<String>,
    session_id: String,
    conversation: Vec<ChatTurn>,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let root = match project_path.filter(|p| !p.trim().is_empty()) {
        Some(path) => PathBuf::from(path)
            .canonicalize()
            .map_err(|e| e.to_string())?,
        None => std::env::current_dir().map_err(|e| e.to_string())?,
    };
    let config = {
        let mut c = state.config.lock().map_err(|e| e.to_string())?;
        if c.providers.is_empty() {
            *c = load_config(&app);
        }
        c.clone()
    };
    let provider = active_provider(&config).ok_or("Add a model provider in Settings first.")?;
    let api_key = provider.api_key.trim();
    let model = provider.model.trim();
    if api_key.is_empty() || model.is_empty() {
        return Err("The active provider needs an API key and model ID. Open Model providers and check both fields.".into());
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("Could not initialize HTTP client: {e}"))?;
    let base = if provider.api_base.trim().is_empty() {
        "https://api.openai.com/v1".to_string()
    } else {
        provider.api_base.trim().trim_end_matches('/').to_string()
    };
    let workspace_tools = json!([
        {"type":"function","function":{"name":"list_files","description":"List workspace directory contents","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
        {"type":"function","function":{"name":"read_file","description":"Read a text file in workspace","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
        {"type":"function","function":{"name":"write_file","description":"Create or replace a workspace file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
        {"type":"function","function":{"name":"git_diff","description":"Show current Git working-tree diff","parameters":{"type":"object","properties":{}}}},
        {"type":"function","function":{"name":"run_command","description":"Execute a shell command in workspace (requires user approval)","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}
    ]);
    let tools = workspace_tools;
    let system_prompt = format!("You are Forge Agent, an autonomous coding assistant with access to the local machine. Use available tools even when no project folder is selected. When no folder is selected, use this default working directory: {}. You may access files outside a selected project when needed. Inspect before editing and explain actions briefly. The application will always ask the user to approve shell commands and file changes. Never claim an action succeeded unless the tool confirms it. Finish with a concise summary.", root.display());
    let mut system_prompt = if config.system_prompt.trim().is_empty() { system_prompt } else { format!("{system_prompt}\n\nAdditional user instructions:\n{}", config.system_prompt.trim()) };
    let enabled_skills: Vec<&AgentSkill> = config.skills.iter().filter(|skill| skill.enabled && !skill.instructions.trim().is_empty()).collect();
    let workspace_skills = load_workspace_skills(&root);
    if !enabled_skills.is_empty() || !workspace_skills.is_empty() {
        system_prompt.push_str("\n\n# Available Agent Skills\nUse a skill when its description matches the task. Skills do not override safety requirements or explicit user instructions.");
        for skill in enabled_skills { system_prompt.push_str(&format!("\n\n## {}\nDescription: {}\n\n{}", skill.name, skill.description, skill.instructions)); }
        for (name, description, instructions) in workspace_skills { system_prompt.push_str(&format!("\n\n## {} (workspace skill)\nDescription: {}\n\n{}", name, description, instructions)); }
    }
    let mut msgs = vec![json!({"role":"system","content":system_prompt})];
    for turn in conversation {
        if turn.role == "user" || turn.role == "assistant" {
            msgs.push(json!({"role":turn.role,"content":turn.content}));
        }
    }
    msgs.push(json!({"role":"user","content":prompt}));
    for step in 0..12 {
        let _ = app.emit(
            "agent-progress",
            format!("{} · thinking / tool cycle {}", provider.name, step + 1),
        );
        let resp = client
            .post(format!("{base}/chat/completions"))
            .bearer_auth(api_key)
            .json(
                &json!({"model":model,"messages":msgs,"tools":tools,"tool_choice":"auto","stream":true}),
            )
            .send()
            .await
            .map_err(|e| format!("{} connection failed ({base}/chat/completions): {e}", provider.name))?;
        let msg = read_stream_response(resp, &app, &session_id).await?;
        msgs.push(msg.clone());
        if let Some(calls) = msg["tool_calls"].as_array() {
            for call in calls {
                let name = call["function"]["name"].as_str().unwrap_or("");
                let args: Value =
                    serde_json::from_str(call["function"]["arguments"].as_str().unwrap_or("{}"))
                        .unwrap_or(json!({}));
                let _ = app.emit("agent-progress", format!("Tool: {name}"));
                let result = tool_exec(name, &args, &root, &app)
                    .unwrap_or_else(|e| format!("Tool error: {e}"));
                msgs.push(json!({"role":"tool","tool_call_id":call["id"],"content":result}));
            }
            continue;
        }
        return Ok(msg["content"]
            .as_str()
            .unwrap_or("Task finished.")
            .to_string());
    }
    Ok("Agent reached the tool-call limit. Review workspace changes.".into())
}
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            status,
            get_config,
            save_config,
            fetch_models,
            choose_project,
            show_git_diff,
            run_task
        ])
        .run(tauri::generate_context!())
        .expect("error running Forge Agent");
}
