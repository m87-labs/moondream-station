import base64
import inspect
import itertools
import shlex
import time
from pathlib import Path
from typing import Any, Callable, Dict, List
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import ANSI
from rich import print as rprint
from rich.panel import Panel

from .core.config import PANEL_WIDTH


class InferenceHandler:
    def __init__(self, repl_session):
        self.repl = repl_session

    def _call_with_first_token_spinner(self, func: Callable, kwargs: Dict[str, Any]):
        result = None
        stream_key = None
        first_token = None
        with self.repl.display.spinner("Waiting for first token..."):
            result = func(**kwargs)
            if isinstance(result, dict):
                for key, value in result.items():
                    if hasattr(value, "__iter__") and hasattr(value, "__next__"):
                        stream_key = key
                        try:
                            first_token = next(value)
                        except StopIteration:
                            first_token = None
                        break

        if stream_key is not None and first_token is not None:
            result[stream_key] = itertools.chain([first_token], result[stream_key])

        return result

    def infer(self, args: List[str]):
        """Run inference: infer <function> [args]"""
        current_model = self.repl.config.get("current_model")
        if not current_model:
            self.repl.display.error(
                "No model selected. Use 'models switch <name>' first."
            )
            return

        backend = self.repl.manifest_manager.get_backend_for_model(current_model)
        if not backend:
            self.repl.display.error("Backend not available for current model")
            return

        model_info = self.repl.models.get_model(current_model)
        if not model_info:
            self.repl.display.error("Model info not found")
            return

        manifest = self.repl.manifest_manager.get_manifest()
        if not manifest or current_model not in manifest.models:
            self.repl.display.error("Model not found in manifest")
            return

        backend_info = manifest.backends.get(manifest.models[current_model].backend)
        if not backend_info:
            self.repl.display.error("Backend info not found")
            return

        if not args:
            rprint(f"[bold]Available functions for {model_info.name}:[/bold]")
            for func in backend_info.functions:
                rprint(f"  {func}")
            rprint(
                "[yellow]Usage:[/yellow] infer <function> [image_path] [additional_args]"
            )
            return

        function_name = args[0]
        if function_name not in backend_info.functions:
            self.repl.display.error(
                f"Function '{function_name}' not available for this model"
            )
            rprint(f"Available functions: {', '.join(backend_info.functions)}")
            return

        if not hasattr(backend, function_name):
            self.repl.display.error(f"Function '{function_name}' not found in backend")
            return

        func = getattr(backend, function_name)
        if not callable(func):
            self.repl.display.error(f"'{function_name}' is not callable")
            return

        try:
            kwargs = self._parse_infer_args(args[1:], function_name)
            kwargs["stream"] = True

            print()
            rprint(f"[bold blue]●[/bold blue] [bold]{function_name}[/bold]")
            self._display_inputs(function_name, args[1:])
            print()

            result = self._call_with_first_token_spinner(func, kwargs)
            self._display_inference_result(result)

        except Exception as e:
            self.repl.analytics.track_error(
                type(e).__name__, str(e), f"direct_inference_{function_name}"
            )
            self.repl.display.error(f"Inference failed: {str(e)}")

    def inference_mode(self, args: List[str]):
        """Enter inference mode for the current model"""
        current_model = self.repl.config.get("current_model")
        if not current_model:
            self.repl.display.error(
                "No model selected. Use 'models switch <name>' first."
            )
            return

        backend = self.repl.manifest_manager.get_backend_for_model(current_model)
        if not backend:
            self.repl.display.error("Backend not available for current model")
            return

        model_info = self.repl.models.get_model(current_model)
        if not model_info:
            self.repl.display.error("Model info not found")
            return

        manifest = self.repl.manifest_manager.get_manifest()
        if not manifest or current_model not in manifest.models:
            self.repl.display.error("Model not found in manifest")
            return

        backend_info = manifest.backends.get(manifest.models[current_model].backend)
        if not backend_info:
            self.repl.display.error("Backend info not found")
            return

        self._enter_inference_mode(model_info, backend_info, backend)

    def _enter_inference_mode(self, model_info, backend_info, backend):
        """Enter dedicated inference mode"""
        self.repl.console.clear()

        combined_content = "Available functions:\n"
        for func_name in backend_info.functions:
            if hasattr(backend, func_name):
                func = getattr(backend, func_name)
                signature = self._get_function_signature(func, func_name)
                line = f"  [bold cyan]{func_name}[/bold cyan] {signature}\n"
                combined_content += line
            else:
                combined_content += (
                    f"  [bold cyan]{func_name}[/bold cyan] <image_path> [args]\n"
                )

        combined_content += "\n[dim]Commands:[/dim]\n"
        combined_content += "  [bold]exit[/bold] - Return to main mode\n"
        combined_content += "  [bold]help[/bold] - Show this help\n"
        combined_content += "  [bold]clear[/bold] - Clear screen"

        inference_panel = Panel(
            combined_content.strip(),
            title="[bold green]● INFERENCE MODE[/bold green]",
            border_style="green",
            width=PANEL_WIDTH,
        )
        self.repl.console.print(inference_panel)
        self.repl.console.print()

        while True:
            try:
                # Create colored prompt using prompt_toolkit with ANSI codes
                colored_prompt = f"\033[32m\033[1minference\033[0m ({model_info.name}) > "
                user_input = prompt(ANSI(colored_prompt)).strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "/exit"]:
                    break
                elif user_input.lower() in ["help", "/help"]:
                    self.repl.console.print(inference_panel)
                    continue
                elif user_input.lower() in ["clear", "/clear"]:
                    self.repl.console.clear()
                    self.repl.console.print(inference_panel)
                    self.repl.console.print()
                    continue

                try:
                    args = shlex.split(user_input)
                    if not args:
                        continue

                    function_name = args[0].lower()
                    if function_name not in backend_info.functions:
                        self.repl.display.error(
                            f"Function '{function_name}' not available"
                        )
                        rprint(
                            f"Available functions: {', '.join(backend_info.functions)}"
                        )
                        continue

                    if not hasattr(backend, function_name):
                        self.repl.display.error(
                            f"Function '{function_name}' not found in backend"
                        )
                        continue

                    func = getattr(backend, function_name)
                    if not callable(func):
                        self.repl.display.error(f"'{function_name}' is not callable")
                        continue

                    kwargs = self._parse_infer_args(args[1:], function_name)
                    kwargs["stream"] = True

                    print()
                    rprint(f"[bold blue]●[/bold blue] [bold]{function_name}[/bold]")
                    self._display_inputs(function_name, args[1:])
                    print()

                    start_time = time.time()
                    try:
                        result = self._call_with_first_token_spinner(func, kwargs)
                        stats = self._display_inference_result(result)
                        if stats:
                            self.repl.analytics.track(
                                "inference_complete",
                                {
                                    "function": function_name,
                                    "duration_ms": round(
                                        (time.time() - start_time) * 1000
                                    ),
                                    "tokens": stats["tokens"],
                                    "tokens_per_sec": stats["tokens_per_sec"],
                                    "model": self.repl.config.get("current_model"),
                                },
                            )
                    except Exception as e:
                        self.repl.analytics.track_error(
                            type(e).__name__, str(e), f"inference_{function_name}"
                        )
                        raise

                except ValueError as e:
                    self.repl.analytics.track_error(
                        "ValueError", str(e), "inference_mode_syntax"
                    )
                    self.repl.display.error(f"Invalid command syntax: {e}")
                except Exception as e:
                    self.repl.analytics.track_error(
                        type(e).__name__, str(e), "inference_mode_general"
                    )
                    self.repl.display.error(f"Inference failed: {str(e)}")

            except KeyboardInterrupt:
                rprint("\n[yellow]Use 'exit' to return to main mode[/yellow]")
            except EOFError:
                break

        self.repl.console.clear()
        self.repl.display.show_banner()
        self.repl._show_startup_info()

    def _display_inputs(self, function_name, args):
        """Display inputs in a clean format"""
        if not args:
            return

        if len(args) >= 1:
            image_path = args[0]
            expanded_path = Path(image_path).expanduser()
            display_path = str(expanded_path) if expanded_path.exists() else image_path
            rprint(f"[dim]Image: {display_path}[/dim]")

        if function_name == "query" and len(args) >= 2:
            question = " ".join(args[1:])
            rprint(f"[dim]Question: {question}[/dim]")
        elif function_name in ["detect", "point"] and len(args) >= 2:
            obj = " ".join(args[1:])
            rprint(f"[dim]Object: {obj}[/dim]")
        elif function_name == "caption" and len(args) >= 2:
            length = args[1]
            rprint(f"[dim]Length: {length}[/dim]")

    def _display_inference_result(self, result):
        """Display inference results with nice formatting"""
        if isinstance(result, dict):
            if result.get("error"):
                self.repl.display.error(f"Inference error: {result['error']}")
                return

            rprint("[dim]Output:[/dim]")

            token_count = 0
            start_time = time.time()

            for key, value in result.items():
                if hasattr(value, "__iter__") and hasattr(value, "__next__"):
                    for token in value:
                        print(token, end="", flush=True)
                        token_count += 1
                    print()

                    duration = time.time() - start_time
                    if duration > 0 and token_count > 0:
                        tokens_per_sec = round(token_count / duration, 1)
                        rprint(
                            f"[dim](Tokens: {token_count}, Tok/s: {tokens_per_sec})[/dim]"
                        )

                    print()
                    return {
                        "tokens": token_count,
                        "tokens_per_sec": (
                            tokens_per_sec if duration > 0 and token_count > 0 else 0
                        ),
                    }

            total_tokens = 0
            for key, value in result.items():
                if key in ["count"]:
                    continue

                if isinstance(value, str):
                    print(value)
                    total_tokens += len(value.split())
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        print(f"  {i+1}. {item}")
                elif isinstance(value, (int, float)):
                    print(f"{key}: {value}")

            if total_tokens > 0:
                duration = time.time() - start_time
                if duration > 0:
                    tokens_per_sec = round(total_tokens / duration, 1)
                    rprint(f"[dim]({tokens_per_sec} tok/s)[/dim]")

            print()

    def _get_function_signature(self, func: Callable, func_name: str) -> str:
        """Get a clean function signature for display"""
        try:
            sig = inspect.signature(func)
            params = []
            seen_object = False

            for param_name, param in sig.parameters.items():
                if param_name.startswith("_") or param_name in ["kwargs", "stream"]:
                    continue

                if param_name == "image_url":
                    params.append("<image_path>")
                elif param_name == "question":
                    if param.default is None:
                        params.append("<question>")
                    else:
                        params.append("[question]")
                elif param_name in ["object", "obj"]:
                    if not seen_object:
                        if param.default is None:
                            params.append("<object>")
                        else:
                            params.append("[object]")
                        seen_object = True
                elif param_name == "length":
                    params.append("\[normal|short|long]")
                else:
                    if (
                        param.default != inspect.Parameter.empty
                        and param.default is not None
                    ):
                        params.append(f"[{param_name}={param.default}]")
                    else:
                        params.append(f"<{param_name}>")

            return " ".join(params) if params else ""

        except Exception:
            if func_name == "caption":
                return "<image_path> [normal|short|long]"
            elif func_name == "query":
                return "<image_path> <question>"
            elif func_name in ["detect", "point"]:
                return "<image_path> <object>"
            else:
                return "<image_path> [args]"

    def _parse_infer_args(
        self, args: List[str], function_name: str = None
    ) -> Dict[str, Any]:
        """Parse inference arguments"""
        kwargs = {}

        if args:
            image_path = args[0]
            expanded_path = Path(image_path).expanduser()
            if expanded_path.exists():
                kwargs["image_url"] = self._encode_image(str(expanded_path))
            else:
                kwargs["image_url"] = image_path

        remaining_args = args[1:]
        if remaining_args:
            if function_name == "caption" and len(remaining_args) == 1:
                length = remaining_args[0].lower()
                if length in ["normal", "short", "long"]:
                    kwargs["length"] = length
            elif len(remaining_args) == 1:
                arg = remaining_args[0]
                if arg.startswith('"') and arg.endswith('"'):
                    kwargs["question"] = arg[1:-1]
                    kwargs["object"] = arg[1:-1]
                else:
                    kwargs["question"] = arg
                    kwargs["object"] = arg
            else:
                text = " ".join(remaining_args)
                kwargs["question"] = text
                kwargs["object"] = text

        return kwargs

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 data URL"""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            encoded = base64.b64encode(image_data).decode()
            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}")
