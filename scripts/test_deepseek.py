#!/usr/bin/env python3
"""Test DeepSeek API key validity."""

import sys
import json
import urllib.request
import urllib.error


def test_deepseek_api(api_key: str) -> bool:
    """Test if DeepSeek API key is valid."""
    url = "https://api.deepseek.com/v1/chat/completions"
    
    data = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5
    }).encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            print(f"✅ API Key is VALID!")
            print(f"   Model: {result.get('model', 'unknown')}")
            print(f"   Response: {result['choices'][0]['message']['content'][:50]}...")
            return True
    except urllib.error.HTTPError as e:
        error_body = json.loads(e.read().decode('utf-8'))
        error_msg = error_body.get('error', {}).get('message', 'Unknown error')
        print(f"❌ API Key is INVALID!")
        print(f"   Error: {error_msg}")
        print(f"   Code: {error_body.get('error', {}).get('code', 'unknown')}")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


if __name__ == "__main__":
    # Try to read from config
    import os
    config_path = os.path.expanduser("~/.nanobot/config.json")
    
    api_key = None
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
                api_key = config.get("memu", {}).get("default", {}).get("apiKey")
                print(f"🔍 Found API key in config: {api_key[:15]}..." if api_key else "❌ No API key in config")
        except Exception as e:
            print(f"❌ Failed to read config: {e}")
    
    if not api_key:
        if len(sys.argv) > 1:
            api_key = sys.argv[1]
        else:
            print("\n💡 Usage: python test_deepseek.py [api_key]")
            print("   Or ensure ~/.nanobot/config.json has memu.default.apiKey")
            sys.exit(1)
    
    print(f"\n🧪 Testing DeepSeek API key...")
    print(f"   Key: {api_key[:15]}...")
    print()
    
    valid = test_deepseek_api(api_key)
    
    if not valid:
        print("\n" + "="*60)
        print("🔧 How to fix:")
        print("="*60)
        print("1. Visit: https://platform.deepseek.com/")
        print("2. Login with your account")
        print("3. Go to 'API Keys' section")
        print("4. Create a new API key")
        print("5. Copy the key (starts with 'sk-')")
        print("6. Update ~/.nanobot/config.json:")
        print()
        print('   "memu": {')
        print('     "default": {')
        print('       "apiKey": "sk-your-new-key-here",')
        print('       "baseUrl": "https://api.deepseek.com/v1",')
        print('       "chatModel": "deepseek-chat"')
        print('     }')
        print('   }')
        print()
        print("7. Restart nanobot: sudo systemctl restart nanobot-agent@root")
    
    sys.exit(0 if valid else 1)
