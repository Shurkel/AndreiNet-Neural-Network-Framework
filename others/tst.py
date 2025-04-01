import os
import sys
import time
import re
import random
import json
from datetime import datetime
from groq import Groq
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

# Available colors for agent differentiation
COLORS = {
    "red": Fore.RED,
    "green": Fore.GREEN, 
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE
}

# Initialize client for conversation responses
client = Groq(
    api_key='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'
)

class Agent:
    """Dynamic agent that can take on any role and personality"""
    
    def __init__(self, role, color, personality=None, expertise=None):
        self.role = role
        self.color = color
        self.personality = personality or "helpful and professional"
        self.expertise = expertise or role
        self.knowledge = []
        self.created_at = datetime.now()
        self.id = f"{role}_{self.created_at.strftime('%H%M%S')}"
    
    def speak(self, message, transcript):
        """Display a message from this agent"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {self.color}{self.role}{Style.RESET_ALL}: {message}"
        print(formatted_message)
        transcript.append({"role": self.role.lower(), "content": message, "timestamp": timestamp})
        time.sleep(0.3)  # Natural conversation pacing
        return message
    
    def think(self, task):
        """Generate an initial response based on the agent's role and personality"""
        try:
            prompt = f"""
            You are {self.role} with expertise in {self.expertise} and a {self.personality} personality.
            Please respond to the following task in your specialized capacity: {task}
            Keep your response concise (under 100 words) and in the first person.
            """
            
            response = client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="deepseek-r1-distill-qwen-32b",
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I'm having trouble processing that request: {str(e)}"
            
    def respond_to_discussion(self, task, discussion_history, transcript):
        """Generate a response based on the ongoing discussion"""
        try:
            # Create a conversation history to provide context
            messages = [
                {"role": "system", "content": f"""
                You are {self.role} with expertise in {self.expertise} and a {self.personality} personality.
                You are participating in a multi-agent discussion to solve a task.
                Respond to the previous messages in a collaborative way.
                Your response should:
                1. Acknowledge what others have said
                2. Add your unique expertise to build on their ideas
                3. Be concise (50-100 words) and in the first person
                4. Stay focused on the task at hand
                """}
            ]
            
            # Add the original task
            messages.append({"role": "user", "content": f"Original task: {task}"})
            
            # Add the discussion history
            for entry in discussion_history:
                if entry["role"] != "system":  # Skip system messages
                    messages.append({
                        "role": "user" if entry["role"] != self.role.lower() else "assistant",
                        "content": f"{entry['role'].upper()}: {entry['content']}"
                    })
            
            # Add a specific prompt to generate the next response
            messages.append({"role": "user", "content": f"Now respond as {self.role} to continue this discussion, building on what others have said."})
            
            response = client.chat.completions.create(
                messages=messages,
                model="deepseek-r1-distill-qwen-32b",
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I need to consider what everyone has said but I'm having some difficulty processing right now: {str(e)}"

class ChiefAgent:
    """Central agent that coordinates the creation and interaction of other agents"""
    
    def __init__(self):
        self.agents = {}
        self.color = Fore.YELLOW
        self.role = "Chief"
    
    def speak(self, message, transcript):
        """Display a message from the chief agent"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {self.color}{self.role}{Style.RESET_ALL}: {message}"
        print(formatted_message)
        transcript.append({"role": self.role.lower(), "content": message, "timestamp": timestamp})
        time.sleep(0.3)
        return message
    
    def extract_json_or_create_agents(self, text):
        """Try multiple methods to extract agent definitions from text"""
        # Method 1: Direct JSON regex
        json_match = re.search(r'\[\s*{.*}\s*\]', text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                json_str = json_str.replace("'", '"')
                return json.loads(json_str)
            except:
                pass
        
        # Method 2: Extract structure from code blocks
        code_blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        for block in code_blocks:
            try:
                # Clean up the block and attempt to parse
                clean_block = block.replace("'", '"')
                clean_block = re.sub(r'(\w+):', r'"\1":', clean_block)
                return json.loads(clean_block)
            except:
                continue
        
        # Method 3: Create agents from text description if JSON fails
        # Look for role: descriptions
        roles = re.findall(r'(?:"|\')?role(?:"|\')?:\s*(?:"|\')?([^"\'}\s,]+)', text)
        expertise = re.findall(r'(?:"|\')?expertise(?:"|\')?:\s*(?:"|\')?([^"\'}\s,]+)', text)
        personalities = re.findall(r'(?:"|\')?personality(?:"|\')?:\s*(?:"|\')?([^"\'}\s,]+)', text)
        
        if roles:
            agents = []
            for i in range(len(roles)):
                agent = {
                    "role": roles[i],
                    "expertise": expertise[i] if i < len(expertise) else roles[i],
                    "personality": personalities[i] if i < len(personalities) else "helpful"
                }
                agents.append(agent)
            return agents
        
        # Last resort: Create default agents based on task content
        return [
            {"role": "Problem Solver", "expertise": "analytical thinking", "personality": "logical"},
            {"role": "Creative Thinker", "expertise": "brainstorming", "personality": "imaginative"}
        ]
    
    def create_agent(self, task, transcript):
        """Dynamically create appropriate agent(s) based on task analysis"""
        try:
            # First try with a more explicit JSON instruction
            prompt = f"""
            Analyze this task: "{task}"
            
            Determine what specialist agents would be needed to solve this collaboratively.
            Return ONLY a JSON array with 2-3 agent definitions.
            
            Each agent should have:
            - role: specific job title
            - expertise: their specialty
            - personality: how they communicate
            
            Return ONLY the JSON array, formatted as:
            [
              {{"role": "role_name", "expertise": "specialty", "personality": "traits"}},
              {{"role": "role_name2", "expertise": "specialty2", "personality": "traits2"}}
            ]
            """
            
            response = client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="deepseek-r1-distill-qwen-32b"
            )
            
            # Extract agent definitions from the response
            agent_text = response.choices[0].message.content
            self.speak(f"Planning team composition based on task requirements...", transcript)
            
            # Use our robust extraction function
            agent_definitions = self.extract_json_or_create_agents(agent_text)
            
            if agent_definitions:
                created_agents = []
                
                for definition in agent_definitions:
                    role = definition.get("role", "Assistant")
                    expertise = definition.get("expertise", role)
                    personality = definition.get("personality", "helpful")
                    
                    # Assign a color that hasn't been used yet if possible
                    available_colors = list(COLORS.values())
                    for existing_agent in self.agents.values():
                        if existing_agent.color in available_colors:
                            available_colors.remove(existing_agent.color)
                    
                    if not available_colors:
                        # If all colors used, pick randomly
                        agent_color = random.choice(list(COLORS.values()))
                    else:
                        agent_color = random.choice(available_colors)
                    
                    # Create the agent
                    new_agent = Agent(role, agent_color, personality, expertise)
                    agent_id = f"{role}_{len(self.agents)}"
                    self.agents[agent_id] = new_agent
                    
                    # Introduce the new agent
                    new_agent.speak(f"I'm now online. As a {expertise} specialist with a {personality} approach, I'm ready to help with this task.", transcript)
                    created_agents.append(new_agent)
                
                return created_agents
            
        except Exception as e:
            self.speak(f"Error during agent creation: {str(e)}. Creating general assistants instead.", transcript)
        
        # Fallback: create general assistants with different specialties
        self.speak("Creating a balanced team of specialists for your task.", transcript)
        
        fallback_agents = []
        
        # Create 2 fallback agents with different personalities
        analyst = Agent("Analyst", Fore.BLUE, "logical and methodical", "analytical thinking")
        self.agents["analyst"] = analyst
        analyst.speak("I'll approach this task with analytical precision and logical reasoning.", transcript)
        fallback_agents.append(analyst)
        
        creative = Agent("Creative", Fore.MAGENTA, "imaginative and innovative", "creative solutions")
        self.agents["creative"] = creative
        creative.speak("I'll explore creative angles and think outside the box for this task.", transcript)
        fallback_agents.append(creative)
        
        return fallback_agents
    
    def solve_task(self, task, transcript):
        """Coordinate agents to solve the given task"""
        self.speak(f"Analyzing your request: '{task}'", transcript)
        self.speak("Creating specialized agents for this task...", transcript)
        
        # Create appropriate agents
        agents = self.create_agent(task, transcript) 
        
        # Collaborative problem solving
        self.speak("Now our agents will collaborate to address your request:", transcript)
        
        # Each agent provides their initial thoughts
        for agent in agents:
            response = agent.think(task)
            agent.speak(response, transcript)
        
        # If multiple agents, facilitate REAL discussion with turn-taking
        if len(agents) > 1:
            self.speak("Let's have our specialists discuss this further in real-time:", transcript)
            
            # Create a discussion history (filtered from transcript)
            # Only include messages from this task onwards
            task_index = 0
            for i, entry in enumerate(transcript):
                if entry.get("content") == task and entry.get("role") == "user":
                    task_index = i
                    break
            
            discussion_history = transcript[task_index:]
            
            # Number of discussion rounds
            num_rounds = 3
            
            # For each round, have each agent respond to the ongoing discussion
            for round_num in range(num_rounds):
                self.speak(f"Discussion round {round_num + 1}/{num_rounds}", transcript) if round_num > 0 else None
                
                # Shuffle the agents to randomize speaking order
                round_agents = agents.copy()
                random.shuffle(round_agents)
                
                for agent in round_agents:
                    # Each agent responds to the discussion so far
                    response = agent.respond_to_discussion(task, discussion_history, transcript)
                    agent.speak(response, transcript)
                    
                    # Update discussion history with this new response
                    discussion_history = transcript[task_index:]
                    
                    # Add a small delay between agent responses
                    time.sleep(0.7)
        
        # Chief summarizes the findings
        self.speak("✨ Summary of our collaborative analysis:", transcript)
        
        try:
            # Generate a concise summary based on the actual discussion that occurred
            # Extract the actual conversation that happened
            conversation_text = ""
            for entry in transcript:
                if entry.get("timestamp", ""):  # Only include timestamped entries (actual spoken lines)
                    conversation_text += f"{entry.get('role').upper()}: {entry.get('content')}\n"
            
            summary_prompt = f"""
            Task: {task}
            
            Here's a transcript of a multi-agent discussion that just occurred:
            {conversation_text[-2000:]}  # Limit to last 2000 chars in case it's too long
            
            Create a concise, helpful summary (under 100 words) of the key insights and conclusions reached.
            Focus on actionable takeaways and areas of consensus/disagreement.
            """
            
            summary = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": summary_prompt},
                ],
                model="deepseek-r1-distill-qwen-32b",
            )
            
            self.speak(summary.choices[0].message.content, transcript)
            
        except Exception as e:
            self.speak(f"In conclusion, our specialists have provided their perspectives on your request. Is there anything specific you'd like us to explore further?", transcript)

def save_transcript(transcript):
    """Save the conversation transcript to a timestamped file."""
    filename = f"chat_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        for msg in transcript:
            role = msg.get("role").upper()
            timestamp = msg.get("timestamp", "")
            content = msg.get("content")
            file.write(f"[{timestamp}] [{role}]: {content}\n\n")
    print(f"Transcript saved to {filename}.")

def divider():
    """Print a nice divider to separate interactions"""
    term_width = os.get_terminal_size().columns
    print("\n" + "─" * term_width + "\n")

def print_help():
    """Display available commands and usage information"""
    help_text = f"""
{Fore.CYAN}┌────────────────────────────────────────────┐
│           {Fore.YELLOW}Dynamic Agent System{Fore.CYAN}            │
└────────────────────────────────────────────┘{Style.RESET_ALL}

{Fore.GREEN}Commands:{Style.RESET_ALL}
  /exit   - Quit the chat
  /clear  - Clear the conversation transcript
  /save   - Save the current conversation transcript
  /help   - Show this help message
  /agents - List all active agents

Simply type any task or question, and our AI Chief will dynamically
create appropriate specialist agents to collaborate on your request.

{Fore.YELLOW}Examples:{Style.RESET_ALL}
  - "I need to understand quantum computing fundamentals"
  - "Help me debug this Python code snippet: [your code]"
  - "Compare different investment strategies for retirement"
  - "Generate creative ideas for a sci-fi story"
"""
    print(help_text)

def main():
    transcript = []
    
    # Create the Chief agent who will coordinate everything
    chief = ChiefAgent()
    
    # Welcome message
    divider()
    chief.speak("Welcome to the Dynamic Agent System! 🤖✨", transcript)
    chief.speak("I'll analyze your tasks and create specialized agents on demand to collaborate on solutions.", transcript)
    divider()
    
    print_help()
    
    while True:
        user_input = input(f"{Fore.WHITE}You{Style.RESET_ALL}: ").strip()
        if not user_input:
            continue
        
        # Command handling
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ["/exit", "/quit"]:
                chief.speak("Goodbye! Have a great day.", transcript)
                break
            elif cmd == "/help":
                print_help()
                continue
            elif cmd == "/clear":
                transcript = []
                chief.agents = {}  # Reset all agents
                chief.speak("Conversation cleared. All agents have been reset.", transcript)
                continue
            elif cmd == "/save":
                save_transcript(transcript)
                continue
            elif cmd == "/agents":
                if chief.agents:
                    chief.speak("Currently active agents:", transcript)
                    for agent_id, agent in chief.agents.items():
                        agent.speak(f"I am a {agent.expertise} specialist with a {agent.personality} approach.", transcript)
                else:
                    chief.speak("No agents are currently active. Submit a task to create some!", transcript)
                continue
            else:
                chief.speak("Unknown command. Type /help for a list of commands.", transcript)
                continue
        
        # Append user's task to transcript
        timestamp = datetime.now().strftime("%H:%M:%S")
        transcript.append({"role": "user", "content": user_input, "timestamp": timestamp})
        
        divider()
        # Chief agent coordinates solving the task with dynamically created agents
        chief.solve_task(user_input, transcript)
        divider()
        
    # Option to save transcript on exit
    save_choice = input(f"{Fore.CYAN}Type /save to save transcript, or press Enter to exit:{Style.RESET_ALL} ").strip().lower()
    if save_choice == "/save":
        save_transcript(transcript)
    print("Session ended.")

if __name__ == "__main__":
    main()