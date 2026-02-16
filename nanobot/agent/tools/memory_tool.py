"""Memory tool for MemU integration."""

from typing import Any

from nanobot.agent.tools.base import Tool


class MemoryRetrieveTool(Tool):
    """Tool to retrieve memories from MemU."""
    
    def __init__(self, memory_adapter: Any):
        self.memory_adapter = memory_adapter
    
    @property
    def name(self) -> str:
        return "retrieve_memory"
    
    @property
    def description(self) -> str:
        return (
            "Retrieve relevant memories from the memory system. "
            "Use this to recall information about the user, past conversations, "
            "important events, or any stored memories. "
            "Call this tool proactively when the user asks about previous topics, "
            "their preferences, or anything that might be in memory."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant memories (e.g., user's name, previous topics, key facts)",
                }
            },
            "required": ["query"],
        }
    
    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        """Execute memory retrieval using memU semantic search."""
        if not self.memory_adapter or not self.memory_adapter.enable_memory:
            return "Memory system is not enabled."

        try:
            import asyncio

            # Get memory status
            status = await self.memory_adapter.memu_status()
            if not status.get("enabled"):
                return "Memory system is currently disabled."

            # Check if MemoryAgent is available for semantic retrieval
            agent = self.memory_adapter._memory_agent
            if not agent:
                return "MemoryAgent not initialized. No memories to retrieve."

            # 方法1: 尝试通过 memory_core 或 core 检索
            try:
                # 尝试不同的 API
                if hasattr(agent, 'memory_core') and hasattr(agent.memory_core, 'retrieve'):
                    result = await asyncio.wait_for(
                        agent.memory_core.retrieve(
                            query=query,
                            user_id="default",
                        ),
                        timeout=10.0,
                    )
                elif hasattr(agent, 'core') and hasattr(agent.core, 'retrieve'):
                    result = await asyncio.wait_for(
                        agent.core.retrieve(
                            query=query,
                            user_id="default",
                        ),
                        timeout=10.0,
                    )
                else:
                    raise AttributeError("No retrieve method found")

                items = result.get("items", [])
                categories = result.get("categories", [])

                if items or categories:
                    lines = []
                    if categories:
                        for cat in categories[:5]:
                            name = cat.get("name", "")
                            summary = cat.get("summary", "")[:200]
                            if summary:
                                lines.append(f"[分类: {name}] {summary}")

                    if items:
                        for item in items[:5]:
                            content = item.get("content", "")[:200]
                            memory_type = item.get("memory_type", "")
                            if content:
                                lines.append(f"[{memory_type}] {content}")

                    if lines:
                        return "找到记忆:\n" + "\n".join(lines[:5])

            except AttributeError:
                pass  # 没有retrieve方法，继续降级方案

            # 方法2: 降级到文件检索
            try:
                memory_dir = self.memory_adapter.workspace / ".memu" / "memory"
                user_dir = memory_dir / "nanobot" / "default:unknown:system"

                if not user_dir.exists():
                    return f"No memories found for query: '{query}'"

                # 读取所有记忆文件
                lines = []
                memory_files = {
                    "profile": "个人档案",
                    "event": "重要事件",
                    "reminder": "提醒事项",
                    "interest": "兴趣爱好",
                    "study": "学习记录",
                    "activity": "活动记录"
                }

                for category, label in memory_files.items():
                    file_path = user_dir / f"{category}.md"
                    if file_path.exists():
                        try:
                            content = file_path.read_text(encoding="utf-8").strip()
                            if content and len(content) > 10:
                                # 取前几行作为摘要
                                content_lines = [l.strip() for l in content.split("\n") if l.strip()]
                                summary = "; ".join(content_lines[:3])
                                lines.append(f"[{label}] {summary}")
                        except Exception:
                            pass

                if not lines:
                    return f"No memories found for query: '{query}'"

                return "找到记忆:\n" + "\n".join(lines[:5])

            except Exception as e:
                return f"Error retrieving from files: {str(e)}"

        except Exception as e:
            return f"Error: {str(e)}"


class MemorySaveTool(Tool):
    """Tool to save important information to MemU."""
    
    def __init__(self, memory_adapter: Any):
        self.memory_adapter = memory_adapter
    
    @property
    def name(self) -> str:
        return "save_memory"
    
    @property
    def description(self) -> str:
        return (
            "Save important information to the memory system. "
            "Use this to remember key facts about the user, their preferences, "
            "important events, or any information that should be recalled in future conversations. "
            "Call this tool proactively when the user shares something important about themselves."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The important information to save (e.g., 'User's name is John, they like Python programming')",
                },
                "category": {
                    "type": "string",
                    "description": "Category of the memory (profile, event, reminder, interest, study, activity)",
                    "enum": ["profile", "event", "reminder", "interest", "study", "activity"],
                }
            },
            "required": ["content", "category"],
        }
    
    async def execute(self, **kwargs: Any) -> str:
        content = kwargs.get("content", "")
        category = kwargs.get("category", "activity")
        """Execute memory saving."""
        if not self.memory_adapter or not self.memory_adapter.enable_memory:
            return "Memory system is not enabled."
        
        try:
            import asyncio
            from datetime import datetime
            
            # For now, write directly to file
            # In the future, this should use MemoryAgent.run()
            memory_dir = self.memory_adapter.workspace / ".memu" / "memory"
            user_dir = memory_dir / "nanobot" / "default:unknown:system"
            user_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = user_dir / f"{category}.md"
            
            # Append to existing file or create new
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = f"\n## {timestamp}\n{content}\n"
            
            if file_path.exists():
                existing = file_path.read_text(encoding="utf-8")
                file_path.write_text(existing + entry, encoding="utf-8")
            else:
                file_path.write_text(f"# {category.capitalize()} Memory\n{entry}", encoding="utf-8")
            
            return f"Successfully saved to {category} memory."
            
        except Exception as e:
            return f"Error saving memory: {str(e)}"
