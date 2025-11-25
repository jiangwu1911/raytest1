#!/usr/bin/env python3
"""
Qwen大模型聊天客户端
使用方式: python qwen_chat.py
"""

import requests
import json
import readline  # 用于命令行历史记录

class QwenChatClient:
    def __init__(self, base_url="http://192.168.1.217:8000", model="qwen-0.5b"):
        self.base_url = base_url
        self.model = model
        self.conversation_history = []
        
    def chat(self, message, temperature=0.7, max_tokens=500):
        """发送消息到Qwen模型并获取回复"""
        
        # 添加用户消息到对话历史
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer fake-key"
                },
                json={
                    "model": self.model,
                    "messages": self.conversation_history,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_reply = result["choices"][0]["message"]["content"]
                
                # 添加助手回复到对话历史
                self.conversation_history.append({"role": "assistant", "content": assistant_reply})
                
                return assistant_reply
            else:
                return f"错误: HTTP {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"请求错误: {e}"
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("对话历史已清空")
    
    def show_history(self):
        """显示对话历史"""
        if not self.conversation_history:
            print("没有对话历史")
            return
            
        print("\n=== 对话历史 ===")
        for i, msg in enumerate(self.conversation_history, 1):
            role = "用户" if msg["role"] == "user" else "助手"
            print(f"{i}. {role}: {msg['content']}")
        print("================\n")

def main():
    client = QwenChatClient()
    
    print("=" * 50)
    print("Qwen大模型聊天客户端")
    print("=" * 50)
    print("命令说明:")
    print("  /clear - 清空对话历史")
    print("  /history - 显示对话历史") 
    print("  /quit 或 /exit - 退出程序")
    print("  /help - 显示帮助")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n您: ").strip()
            
            if not user_input:
                continue
                
            # 处理命令
            if user_input.lower() in ['/quit', '/exit']:
                print("再见！")
                break
            elif user_input.lower() == '/clear':
                client.clear_history()
                continue
            elif user_input.lower() == '/history':
                client.show_history()
                continue
            elif user_input.lower() == '/help':
                print("命令说明:")
                print("  /clear - 清空对话历史")
                print("  /history - 显示对话历史") 
                print("  /quit 或 /exit - 退出程序")
                print("  /help - 显示帮助")
                continue
            
            # 发送消息
            print("思考中...", end="", flush=True)
            reply = client.chat(user_input)
            print("\r助手: " + reply)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")

if __name__ == "__main__":
    main()
