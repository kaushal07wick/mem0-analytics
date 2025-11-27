from openai import OpenAI
from mem0 import Memory
import os
import sys

# === Environment Check ===
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or not api_key.strip():
    print("\n❌ ERROR: OPENAI_API_KEY environment variable not found.")
    print("➡️  Set it before running this script:")
    print("   export OPENAI_API_KEY=sk-your-key-here   (Linux/macOS)")
    print("   setx OPENAI_API_KEY sk-your-key-here     (Windows PowerShell)")
    sys.exit(1)
else:
    print("✅ OPENAI_API_KEY loaded successfully from environment.\n")

openai_client = OpenAI()
memory = Memory()

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # retrieve relevant memories
    relevant_memories = memory.search(query=message, user_id=user_id, limit=2)
    memories_str = "\n".join(f" - {entry['memory']}" for entry in relevant_memories["results"])


    # generate assistant response
    system_prompt = f"You are a critique, where the user will post some questions and you should go deep into psychology and answer them in a philospohical way. \nUser Memories: \n{memories_str}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    response = openai_client.chat.completions.create(model="gpt-4.1-nano-2025-04-14", messages=messages)
    assistant_response = response.choices[0].message.content

    # create new memories from the conversation
    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)

    return assistant_response

def main():
    print("chat with ai (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")

if __name__ == "__main__":
    main()