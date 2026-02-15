# Nanobot Embedding 配置指南

## 配置位置

在 `~/.nanobot/config.json` 中的 `memu.embedding` 部分：

```json
{
  "memu": {
    "enabled": true,
    "default": {
      "provider": "openai",
      "baseUrl": "https://api.deepseek.com/v1",
      "apiKey": "sk-your-deepseek-key",
      "chatModel": "deepseek-chat"
    },
    "embedding": {
      "provider": "openai",
      "baseUrl": "https://api.siliconflow.cn/v1",
      "apiKey": "sk-your-siliconflow-key",
      "embedModel": "BAAI/bge-m3",
      "clientBackend": "sdk"
    }
  }
}
```

## 支持的 Embedding 提供商

### 1. SiliconFlow（推荐，国内可用）
```json
"embedding": {
  "provider": "openai",
  "baseUrl": "https://api.siliconflow.cn/v1",
  "apiKey": "sk-your-siliconflow-api-key",
  "embedModel": "BAAI/bge-m3",
  "clientBackend": "sdk"
}
```
**可用模型**：`BAAI/bge-m3`, `netease-youdao/bce-embedding-base_v1`

### 2. OpenAI（需要海外访问）
```json
"embedding": {
  "provider": "openai",
  "baseUrl": "https://api.openai.com/v1",
  "apiKey": "sk-your-openai-api-key",
  "embedModel": "text-embedding-3-small",
  "clientBackend": "sdk"
}
```
**可用模型**：`text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`

### 3. Azure OpenAI（企业环境）
```json
"embedding": {
  "provider": "openai",
  "baseUrl": "https://your-resource.openai.azure.com/openai/deployments/your-deployment",
  "apiKey": "your-azure-api-key",
  "embedModel": "text-embedding-3-small",
  "clientBackend": "sdk"
}
```

### 4. 禁用 Embedding（最简单）
如果不配置 `embedding` 部分，系统将自动禁用语义搜索，但记忆功能仍然可用。

```json
{
  "memu": {
    "enabled": true,
    "default": {
      "provider": "openai",
      "baseUrl": "https://api.deepseek.com/v1",
      "apiKey": "sk-your-deepseek-key",
      "chatModel": "deepseek-chat"
    }
  }
}
```

## 如何获取 API Key

### SiliconFlow
1. 访问 https://cloud.siliconflow.cn/
2. 注册账号
3. 进入控制台 → API 密钥
4. 创建新的 API Key

### OpenAI
1. 访问 https://platform.openai.com/
2. 注册账号并绑定支付方式
3. 进入 API Keys 页面
4. 创建新的 Secret Key

## 故障排除

### 错误："Model does not exist"
**原因**：配置的模型在提供商处不可用
**解决**：检查模型名称是否正确，或更换提供商

### 错误："Authentication Fails"
**原因**：API Key 无效或过期
**解决**：检查 API Key 是否正确，或重新生成

### 错误："Failed to initialize embedding client"
**原因**：环境变量配置错误
**解决**：重启 nanobot 服务：`sudo systemctl restart nanobot-agent@root`

## 测试配置

```bash
# 1. 检查配置文件语法
python -c "import json; json.load(open('/root/.nanobot/config.json'))"

# 2. 重启服务
sudo systemctl restart nanobot-agent@root

# 3. 查看日志确认 embedding 模型
sudo journalctl -u nanobot-agent@root -f | grep -E "(Embedding model|embedding)"
```
