from ast import Mod
import base64
from calendar import c
from contextlib import contextmanager
import inspect
from json import tool
from operator import call
import os
import json
import re
from typing import Any, Callable, Generator
import uuid
from attr import dataclass
from dotenv import load_dotenv
from numpy import full
from openai import OpenAI, omit
import gradio as gr
from datetime import datetime

def today_date():
    """Tool function to return today's date"""
    return datetime.now().strftime("%Y-%m-%d")

@dataclass
class ToolResult:
    content_for_model: str
    content_type:str = "text"
    content:Any

class Models:
    IMAGES = "grok-imagine-image"
    QUESTIONS = "grok-4-1-fast-reasoning"
    COMPLEX_QUESTIONS = "grok-4.3"

class ModelContext:
    _instance:"ModelContext" = None

    def __init__(self, client:OpenAI, image_path:str):
        self.model_name = Models.QUESTIONS
        self._client = client
        self._image_path = image_path

    @classmethod
    def create(cls, client:OpenAI, image_path:str):
        if cls._instance is None:
            cls._instance = cls(client, image_path)
        return cls._instance
    
    @classmethod
    def current(cls):
        if cls._instance:
            return cls._instance
        
        raise Exception("ModelContext has not been initialized. Call ModelContext.create(client, image_path) first.")

    @property
    def model_name(self)->str:
        return self._model_name

    @model_name.setter
    def model_name(self, value:str):
        self._validate_model(value)
        self._model_name = value
    
    def _validate_model(self, model_name:str):
        if model_name not in [Models.IMAGES, Models.QUESTIONS, Models.COMPLEX_QUESTIONS]:
            raise ValueError(f"Invalid model name: {model_name}")
        
    @property
    def client(self)->OpenAI:
        return self._client
    
    @property
    def image_path(self)->str:
        return self._image_path
    
    @contextmanager
    def use_model(self, model_name:str):
        """Context manager to temporarily switch models."""
        self._validate_model(model_name)
        old_model = self.model_name
        self.model_name = model_name
        try:
            yield
        finally:
            self.model_name = old_model

class Tools:
    def __init__(self):
        self.tools = {}

    def register_tool(self, func:Callable, description:str):
        """Registers a function as a tool for the LLM to use.

        Parameter types are derived from function annotations, and descriptions are derived from the function docstrings
        Args:
            func (Callable): The function to register as a tool.
            description (str): A brief description of what the tool does.
        """
        parameters = {}
        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            parameters[param_name] = {"type": get_property_type(param.annotation, True), "description": get_description(func.__doc__, param_name)}
        self.tools[func.__name__] = {"function": func,
                            "json": {
                                "name": func.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                    },
                    "required": [
                        name for name, param in sig.parameters.items()
                        if param.default is inspect.Parameter.empty 
                        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                    ],
                    "additional_properties": False,
                    }
                }

    def remove_tool(self, func:Callable):
        """Removes a tool from the registry."""
        if func.__name__ in self.tools:
            del self.tools[func.__name__]

    def handle_tool_calls(self, message)->tuple[list[dict[str, Any]], list[ToolResult|None]]:
        """Handles tool calls from the model by executing the corresponding functions and returning their results."""
        responses = []
        tool_results:list[ToolResult|None] = []
        for tool_call in message.tool_calls:
            print(f"Got tool call {tool_call}")
            func = self.tools.get(tool_call.function.name, {})
            if func:
                arguments = json.loads(tool_call.function.arguments)
                print(f"Calling function {func['function']} with arguments {arguments}")
                result = func['function'](**arguments)
                print(f"Got result {result} from tool call")
                
                result_for_model = result
                if isinstance(result, ToolResult):
                    result_for_model = result.content_for_model
                    tool_results.append(result.content)
                else:
                    tool_results.append(None)

                responses.append({
                    "role": "tool",
                    "content": result_for_model or "",
                    "tool_call_id": tool_call.id
                })
        return responses, tool_results

    def get_tools_for_model(self):
        return [{"type": "function", "function": self.tools[t]["json"]} for t in self.tools]    

def init_api(api_key_name:str, api_url:str)->OpenAI:
    load_dotenv()
    api_key = os.getenv(api_key_name)
    return OpenAI(api_key=api_key, base_url=api_url)


def get_property_type(annotation, for_xai: bool):
    """Helper function for Tools class to get property types."""
    if for_xai:
        if annotation == str or annotation == inspect.Parameter.empty:
            return "string"
    return annotation.__name__ if annotation != inspect.Parameter.empty else "str"


def get_description(docstring, param_name):
    """A naive way to extract parameter descriptions from a docstring."""
    if not docstring:
        return ""
    lines = docstring.splitlines()
    for line in lines:
        line = line.strip()
        if param_name in line and ':' in line:
            return line.partition(':')[2].strip()
    return ""


class ChatInterface:
    SYSTEM_MESSAGE = "You are a helpful assistant. Only answer factually, and if you do not know the answer don't make anything up."
    
    def __init__(self, client: OpenAI, tools: Tools):
        self.client = client
        self.tools = tools
        self.chat_history: list[dict[str, str]] = []
        
    def _get_model_for_choice(self, choice: str) -> str:
        """Map dropdown choice to appropriate model."""
        if choice == "Question":
            return Models.QUESTIONS
        elif choice == "Complex Question":
            return Models.COMPLEX_QUESTIONS
        elif choice == "Generate Image":
            return Models.QUESTIONS
        return Models.QUESTIONS
    
    def _get_system_message_for_choice(self, choice:str)->str:
        """Get system message based on choice."""
        if choice == "Generate Image":
            return """"You are an assistant that helps generate images based on user requests.
             The user will provide a description of the image they want and you will help them refine that description if needed, and then call the image generation tool with the final prompt.
             The user must not ask for an image that contains nudity, suggestive content, or graphic violence. However, acts of affection (e.g. hugging, kissing) and display of weapons (e.g swords, guns, knives) are acceptable.
             If the user requests an image that violates the content guidelines, let the user know tht you cannot generate the image because it violates the content guidelines and what part of their request violates the guidelines.
                Always follow the guidelines and never generate content that violates them.
             If the use asks for an image, but does not provide enough details, tell the user thay they need to supply additional details and give suggestion a suggestion based on the user's prompt that contains additional details.

             Examples:
                User: I want a picture of a dog.
                Assistant: Can you provide more details about the dog picture you want? For example, what breed of dog, what setting or background, any specific colors or actions you want the dog to be doing?

                User: I would like a picture of a man with dark hair and skin wearing armor and holding a sword in an outstreached hand. The man is standing on a desolate battlefield with smoke in the background.
                Assistant: Here is the image based on your request.

                User: I want a picture of a man with his head cut off.
                Assistant: I'm sorry, but I cannot generate that image because it violates the content guidelines regarding graphic violence. Specifically, the request for a picture of a man with his head cut off is not something I can assist with. Please let me know if you have another image request that follows the guidelines.

                User: I want a picture of a woman in a bikini.
                Assistant: I'm sorry, but I cannot generate that image because it violates the content guidelines regarding suggestive content. Specifically, the request for a pictures of people in bikinis is not something I can assist with. Please let me know if you have another image request that follows the guidelines.
             """
        return self.SYSTEM_MESSAGE
    def _collect_stream(self, stream):
        """Collect all chunks from a stream, yielding text content and accumulating tool calls."""
        full_content = ""
        
        for chunk in stream:
            delta = chunk.choices[0].delta
            
            # Collect text content
            if delta.content:
                token = delta.content
                full_content += token
                yield ("text", token)
            
            # Collect tool calls (they come in parts across chunks)
            if delta.tool_calls:
                tool_responses, tool_results = self.tools.handle_tool_calls(delta)
                yield ("tool_call", tool_responses)
                yield ("tool_result", tool_results)
    
    def chat(self, message: str, chat_history: list[dict[str, str]], choice: str)->Generator[Any, Any, Any]:
        """Stream chat responses token by token with proper tool handling."""
        if not message:
            return chat_history, None
        
        # Add user message to history
        self.chat_history.append({"role": "user", "content": message})
        local_chat_history = [c for c in self.chat_history]  # Create a local copy for this interaction
        chat_history.append([message, ""])
        
        model = self._get_model_for_choice(choice)
        is_image_mode = choice == "Generate Image"
        
        try:
            # Process in a loop to handle tool calls and follow-ups
            while True:
                # Build messages with system context
                messages = [{"role": "system", "content": self.SYSTEM_MESSAGE}]
                messages.extend(self.chat_history)
                
                # Create streaming response
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=self.tools.get_tools_for_model() if self.tools.tools else omit,
                    stream=True,
                )
                
                # Collect and stream all chunks
                full_response = ""    
                tool_calls = []        
                for item_type, item_data in self._collect_stream(stream):
                    if item_type == "text":
                        full_response += item_data
                        yield local_chat_history + [{"role": "assistant", "content": full_response or ""}], None
                    elif item_type == "tool_call":
                        tool_calls.extend(item_data)
                    elif item_type == "tool_result" and is_image_mode:
                        for image_data in [i for i in item_data if i and i.content_type == "image"]:
                            yield local_chat_history + [{"role": "assistant", "content": full_response or ""}], image_data
                
                # Add assistant response to history
                if full_response:
                    self.chat_history.append({"role": "assistant", "content": full_response or ""})
                if tool_calls:
                    self.chat_history.extend(tool_calls)
                    continue
                
                break
                
        except Exception as e:
            print(f"Error in chat: {e}")
            # Show generic error without details
            error_msg = "I encountered an error processing your request. Please try again."
            chat_history[-1][1] = error_msg
            self.chat_history.append({"role": "assistant", "content": error_msg})
            yield self.chat_history, None
    
    def _extract_image_from_response(self, response: str):
        """Extract image data from response (URL, file path, or base64)."""
        # Simple extraction - look for common URL patterns or file paths
        
        # Look for URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, response)
        if urls:
            return urls[0]
        
        # Look for file paths
        path_pattern = r'[/\\][\w/\\.-]*\.(?:png|jpg|jpeg|gif|webp)'
        paths = re.findall(path_pattern, response)
        if paths:
            return paths[0]
        
        # Return None if no image found
        return None
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        return []


def launch_app(client: OpenAI, tools: Tools):
    """Launch the Gradio Blocks interface."""
    chat_interface = ChatInterface(client, tools)
    
    with gr.Blocks(title="AI Assistant") as demo:
        gr.Markdown("# AI Assistant")
        
        with gr.Row():
            choice = gr.Dropdown(
                choices=["Question", "Complex Question", "Generate Image"],
                value="Question",
                label="Select Mode",
                interactive=True
            )
        
        chatbot = gr.Chatbot(
            label="Chat",
            type="messages",
            height=400
        )
        
        image_output = gr.Image(
            label="Generated Image",
            visible=False,
            type="filepath"
        )
        
        with gr.Row():
            message_input = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
            )
        
        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")
        
        def on_choice_change(selected_choice):
            """Update image visibility based on choice."""
            return gr.update(visible=(selected_choice == "Generate Image"))
        
        choice.change(
            fn=on_choice_change,
            inputs=choice,
            outputs=image_output
        )
        
        def handle_submit(message, chat_history, selected_choice):
            """Handle message submission."""
            if not message:
                return chat_history, "", None
            
            for updated_history, image_data in chat_interface.chat(message, chat_history, selected_choice):
                yield updated_history, "", image_data
        
        def handle_clear():
            """Handle clear button."""
            cleared_history = chat_interface.clear_history()
            return cleared_history, "", None
        
        submit_event = {"fn": handle_submit,
            "inputs": [message_input, chatbot, choice],
            "outputs": [chatbot, message_input, image_output]}
        
        # Submit on button click
        submit_btn.click(
            **submit_event
        )
        
        # Submit on Enter key (textbox submission)
        message_input.submit(**submit_event)
        
        
        # Clear button
        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot, message_input, image_output]
        )
    
    return demo

def generate_image(prompt:str, model:str, client:OpenAI, image_path:str)->tuple[str, bytes]:
    """Example function to generate an image using the API."""
    image_system_prompt = "Generate an image basedo on the request below. The generaged image must not include any nudity, suggestive content, or graphic violence. If requested, acts of affection (e.g. hugging, kissing) and display of weapons (e.g swords, guns, knives) is acceptable. If the requested picture does not meet the guidelines generate a picture of a peaceful landscape instead. Always follow the guidelines and never generate content that violates them.\n\nRequest:"
    image_response = client.images.generate(
            model=model,
            prompt=f"{image_system_prompt}\n\n{prompt}",
            size=omit,
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    image_file = os.path.join(image_path, f"{uuid.uuid4()}.png")
    with open(image_file, "wb") as f:
        f.write(image_data)
    
    print(f"Generated image saved to {image_file}")

    return (image_file, image_data)

def generate_image_tool(prompt:str)->ToolResult:
    """Tool function to generate an image."""
    model_context = ModelContext.current()
    with model_context.use_model(Models.IMAGES):
        _, image_data = generate_image(prompt, model_context.model_name, model_context.client, model_context.image_path)
    return ToolResult(content_for_model="Generated an image.", content=image_data)

# Example usage (uncomment to run)
if __name__ == "__main__":
    load_dotenv()
    xai_api_key = os.getenv('XAI_API_KEY')
    client = init_api("XAI_API_KEY", "https://api.x.ai/v1")
    ModelContext.create(client, r"C:\Users\jordan-dev\model_output\images")
    # tools = Tools()
    # tools.register_tool(today_date, "Get today's date in YYYY-MM-DD format")
    # app = launch_app(client, tools)
    # app.launch()

    generate_image("A picture of a 4 year old girl in a pink dress running through a field of long grass while laughing. She has curly, brown hair, and her skin is tan from being in the sun. She is holding a flower in one hand.",
                   Models.IMAGES, client, r"C:\Users\jordan-dev\model_output\images")


