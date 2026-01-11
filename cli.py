"""CLI enhancements for Social Media Publisher.

This module provides a modern command-line interface using the Click
library patterns (pure Python implementation for no extra dependencies).

Features:
- Command completion support
- Progress bars for long operations
- Dry-run mode for testing
- Verbose/quiet modes for output control
- Colored output for better readability
- Help text for all commands

Example:
    cli = CLI()
    cli.run(["--dry-run", "search", "--count", "5"])
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from test_framework import TestSuite


# =============================================================================
# Output Verbosity
# =============================================================================


class Verbosity(Enum):
    """Output verbosity levels."""

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


# =============================================================================
# Console Colors
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"

    @classmethod
    def enabled(cls) -> bool:
        """Check if colors should be enabled."""
        # Disable on Windows unless ANSI is supported
        if sys.platform == "win32":
            try:
                import os

                return os.environ.get("TERM") is not None or os.environ.get("ANSICON") is not None
            except Exception:
                return False
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def colorize(text: str, color: str) -> str:
    """Colorize text if colors are enabled."""
    if Colors.enabled():
        return f"{color}{text}{Colors.RESET}"
    return text


# =============================================================================
# Progress Bar
# =============================================================================


@dataclass
class ProgressBar:
    """Simple progress bar for terminal output."""

    total: int
    width: int = 40
    prefix: str = ""
    suffix: str = ""
    fill: str = "█"
    empty: str = "░"
    current: int = 0
    start_time: float = field(default_factory=time.time)
    _last_render: str = ""

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current = min(self.current + n, self.total)
        self._render()

    def set(self, value: int) -> None:
        """Set progress to specific value."""
        self.current = min(max(0, value), self.total)
        self._render()

    def _render(self) -> None:
        """Render the progress bar."""
        if self.total == 0:
            percent = 100
        else:
            percent = int(100 * self.current / self.total)

        filled = int(self.width * self.current / self.total) if self.total > 0 else self.width
        bar = self.fill * filled + self.empty * (self.width - filled)

        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.current > 0 and self.current < self.total:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f" ETA: {eta:.1f}s"
        elif self.current >= self.total:
            eta_str = f" Done in {elapsed:.1f}s"
        else:
            eta_str = ""

        render = f"\r{self.prefix}|{bar}| {percent}% ({self.current}/{self.total}){self.suffix}{eta_str}"

        # Only print if changed
        if render != self._last_render:
            sys.stdout.write(render)
            sys.stdout.flush()
            self._last_render = render

    def finish(self) -> None:
        """Complete the progress bar."""
        self.current = self.total
        self._render()
        print()  # New line


def progress_bar(items: list[Any], desc: str = "") -> ProgressBar:
    """Create a progress bar for iterating over items."""
    bar = ProgressBar(total=len(items), prefix=f"{desc} " if desc else "")
    return bar


# =============================================================================
# Spinner
# =============================================================================


@dataclass
class Spinner:
    """Simple spinner for indeterminate progress."""

    message: str = "Loading"
    frames: list[str] = field(
        default_factory=lambda: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    )
    _frame: int = 0
    _active: bool = False

    def step(self) -> str:
        """Get next spinner frame."""
        frame = self.frames[self._frame % len(self.frames)]
        self._frame += 1
        return f"\r{frame} {self.message}"

    def render(self) -> None:
        """Render current spinner state."""
        sys.stdout.write(self.step())
        sys.stdout.flush()

    def done(self, message: str = "Done") -> None:
        """Complete the spinner with a message."""
        sys.stdout.write(f"\r✓ {message}\n")
        sys.stdout.flush()

    def fail(self, message: str = "Failed") -> None:
        """Fail the spinner with a message."""
        sys.stdout.write(f"\r✗ {message}\n")
        sys.stdout.flush()


# =============================================================================
# Command Definition
# =============================================================================


@dataclass
class Option:
    """Command-line option definition."""

    name: str
    short: str | None = None
    help: str = ""
    type: type = str
    default: Any = None
    required: bool = False
    is_flag: bool = False

    def matches(self, arg: str) -> bool:
        """Check if argument matches this option."""
        if arg == f"--{self.name}":
            return True
        if self.short and arg == f"-{self.short}":
            return True
        return False


@dataclass
class Argument:
    """Positional argument definition."""

    name: str
    help: str = ""
    type: type = str
    required: bool = True
    default: Any = None


@dataclass
class Command:
    """CLI command definition."""

    name: str
    help: str
    handler: Callable[..., Any]
    options: list[Option] = field(default_factory=list)
    arguments: list[Argument] = field(default_factory=list)
    subcommands: list["Command"] = field(default_factory=list)

    def format_help(self) -> str:
        """Format help text for this command."""
        lines = [
            f"{colorize(self.name, Colors.BOLD)} - {self.help}",
            "",
        ]

        if self.arguments:
            lines.append("Arguments:")
            for arg in self.arguments:
                req = "(required)" if arg.required else f"(default: {arg.default})"
                lines.append(f"  {arg.name:15} {arg.help} {req}")
            lines.append("")

        if self.options:
            lines.append("Options:")
            for opt in self.options:
                short = f"-{opt.short}, " if opt.short else "    "
                flag_type = "" if opt.is_flag else f" <{opt.type.__name__}>"
                lines.append(f"  {short}--{opt.name}{flag_type:10} {opt.help}")
            lines.append("")

        if self.subcommands:
            lines.append("Subcommands:")
            for sub in self.subcommands:
                lines.append(f"  {sub.name:15} {sub.help}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# CLI Context
# =============================================================================


@dataclass
class CLIContext:
    """Context passed to command handlers."""

    verbose: Verbosity = Verbosity.NORMAL
    dry_run: bool = False
    color: bool = True
    args: dict[str, Any] = field(default_factory=dict)

    def echo(self, message: str, level: Verbosity = Verbosity.NORMAL) -> None:
        """Print message if verbosity allows."""
        if level.value <= self.verbose.value:
            print(message)

    def info(self, message: str) -> None:
        """Print info message."""
        if self.verbose.value >= Verbosity.NORMAL.value:
            prefix = colorize("INFO", Colors.BLUE) if self.color else "INFO"
            print(f"[{prefix}] {message}")

    def success(self, message: str) -> None:
        """Print success message."""
        if self.verbose.value >= Verbosity.NORMAL.value:
            prefix = colorize("OK", Colors.GREEN) if self.color else "OK"
            print(f"[{prefix}] {message}")

    def warning(self, message: str) -> None:
        """Print warning message."""
        if self.verbose.value >= Verbosity.QUIET.value:
            prefix = colorize("WARN", Colors.YELLOW) if self.color else "WARN"
            print(f"[{prefix}] {message}")

    def error(self, message: str) -> None:
        """Print error message."""
        prefix = colorize("ERROR", Colors.RED) if self.color else "ERROR"
        print(f"[{prefix}] {message}", file=sys.stderr)

    def debug(self, message: str) -> None:
        """Print debug message."""
        if self.verbose.value >= Verbosity.DEBUG.value:
            prefix = colorize("DEBUG", Colors.DIM) if self.color else "DEBUG"
            print(f"[{prefix}] {message}")


# =============================================================================
# Argument Parser
# =============================================================================


class ArgumentParser:
    """Simple argument parser for CLI commands."""

    def __init__(self, command: Command) -> None:
        """Initialize parser for a command."""
        self.command = command

    def parse(self, args: list[str]) -> tuple[dict[str, Any], list[str]]:
        """Parse arguments and return (parsed_options, remaining)."""
        parsed: dict[str, Any] = {}
        remaining: list[str] = []
        i = 0

        # Set defaults
        for opt in self.command.options:
            if opt.is_flag:
                parsed[opt.name] = False
            else:
                parsed[opt.name] = opt.default

        for arg in self.command.arguments:
            parsed[arg.name] = arg.default

        # Parse arguments
        arg_index = 0
        while i < len(args):
            arg = args[i]

            # Check for options
            option_found = False
            for opt in self.command.options:
                if opt.matches(arg):
                    option_found = True
                    if opt.is_flag:
                        parsed[opt.name] = True
                    else:
                        if i + 1 < len(args):
                            i += 1
                            value = args[i]
                            if opt.type is int:
                                parsed[opt.name] = int(value)
                            elif opt.type is float:
                                parsed[opt.name] = float(value)
                            elif opt.type is bool:
                                parsed[opt.name] = value.lower() in ("true", "1", "yes")
                            else:
                                parsed[opt.name] = value
                    break

            if not option_found:
                if arg.startswith("-"):
                    remaining.append(arg)
                else:
                    # Positional argument
                    if arg_index < len(self.command.arguments):
                        arg_def = self.command.arguments[arg_index]
                        if arg_def.type is int:
                            parsed[arg_def.name] = int(arg)
                        elif arg_def.type is float:
                            parsed[arg_def.name] = float(arg)
                        else:
                            parsed[arg_def.name] = arg
                        arg_index += 1
                    else:
                        remaining.append(arg)

            i += 1

        # Check required arguments
        for arg in self.command.arguments:
            if arg.required and parsed.get(arg.name) is None:
                raise ValueError(f"Missing required argument: {arg.name}")

        return parsed, remaining


# =============================================================================
# CLI Application
# =============================================================================


class CLI:
    """Command-line interface application."""

    def __init__(self, name: str = "smp", version: str = "1.0.0") -> None:
        """Initialize CLI application."""
        self.name = name
        self.version = version
        self.commands: dict[str, Command] = {}
        self.global_options = [
            Option("help", "h", "Show help message", is_flag=True),
            Option("version", "V", "Show version", is_flag=True),
            Option("verbose", "v", "Verbose output", is_flag=True),
            Option("quiet", "q", "Quiet output", is_flag=True),
            Option("dry-run", "n", "Dry run mode (no changes)", is_flag=True),
            Option("no-color", None, "Disable colored output", is_flag=True),
        ]

    def add_command(self, command: Command) -> None:
        """Add a command to the CLI."""
        self.commands[command.name] = command

    def get_completions(self, partial: str) -> list[str]:
        """Get command completions for partial input."""
        completions = []

        # Complete command names
        for name in self.commands:
            if name.startswith(partial):
                completions.append(name)

        # Complete global options
        for opt in self.global_options:
            if f"--{opt.name}".startswith(partial):
                completions.append(f"--{opt.name}")
            if opt.short and f"-{opt.short}".startswith(partial):
                completions.append(f"-{opt.short}")

        return completions

    def format_help(self) -> str:
        """Format main help text."""
        lines = [
            colorize(f"{self.name} v{self.version}", Colors.BOLD),
            "",
            "Usage: {name} [options] <command> [args...]".format(name=self.name),
            "",
            "Global Options:",
        ]

        for opt in self.global_options:
            short = f"-{opt.short}, " if opt.short else "    "
            lines.append(f"  {short}--{opt.name:15} {opt.help}")

        lines.extend(["", "Commands:"])

        for name, cmd in sorted(self.commands.items()):
            lines.append(f"  {name:15} {cmd.help}")

        lines.extend(
            [
                "",
                f'Use "{self.name} <command> --help" for command-specific help.',
            ]
        )

        return "\n".join(lines)

    def run(self, args: list[str] | None = None) -> int:
        """Run the CLI with given arguments."""
        if args is None:
            args = sys.argv[1:]

        # Parse global options first
        context = CLIContext()
        remaining_args = []
        i = 0

        while i < len(args):
            arg = args[i]
            handled = False

            for opt in self.global_options:
                if opt.matches(arg):
                    handled = True
                    if opt.name == "help":
                        print(self.format_help())
                        return 0
                    elif opt.name == "version":
                        print(f"{self.name} v{self.version}")
                        return 0
                    elif opt.name == "verbose":
                        context.verbose = Verbosity.VERBOSE
                    elif opt.name == "quiet":
                        context.verbose = Verbosity.QUIET
                    elif opt.name == "dry-run":
                        context.dry_run = True
                    elif opt.name == "no-color":
                        context.color = False
                    break

            if not handled:
                remaining_args.append(arg)

            i += 1

        # No command specified
        if not remaining_args:
            print(self.format_help())
            return 0

        # Find and run command
        cmd_name = remaining_args[0]
        cmd_args = remaining_args[1:]

        if cmd_name not in self.commands:
            context.error(f"Unknown command: {cmd_name}")
            context.echo(f'Use "{self.name} --help" for available commands.')
            return 1

        command = self.commands[cmd_name]

        # Check for command help
        if "--help" in cmd_args or "-h" in cmd_args:
            print(command.format_help())
            return 0

        # Parse command arguments
        try:
            parser = ArgumentParser(command)
            parsed, _ = parser.parse(cmd_args)
            context.args = parsed
        except ValueError as e:
            context.error(str(e))
            return 1

        # Run command
        try:
            if context.dry_run:
                context.warning("DRY RUN MODE - no changes will be made")

            result = command.handler(context)
            return result if isinstance(result, int) else 0
        except Exception as e:
            context.error(f"Command failed: {e}")
            if context.verbose == Verbosity.DEBUG:
                import traceback

                traceback.print_exc()
            return 1


# =============================================================================
# Default Commands (examples)
# =============================================================================


def cmd_search(ctx: CLIContext) -> int:
    """Search for stories."""
    count = ctx.args.get("count", 5)

    ctx.info(f"Searching for {count} stories...")

    if ctx.dry_run:
        ctx.warning("Would search for stories (dry run)")
        return 0

    # Simulate search with progress bar
    bar = ProgressBar(total=count, prefix="Searching ")
    for i in range(count):
        time.sleep(0.1)  # Simulate work
        bar.update()
    bar.finish()

    ctx.success(f"Found {count} stories")
    return 0


def cmd_publish(ctx: CLIContext) -> int:
    """Publish a story."""
    story_id = ctx.args.get("story_id")
    force = ctx.args.get("force", False)

    if not story_id:
        ctx.error("No story ID provided")
        return 1

    ctx.info(f"Publishing story {story_id}...")

    if force:
        ctx.warning("Force mode enabled - skipping validation")

    if ctx.dry_run:
        ctx.warning(f"Would publish story {story_id} (dry run)")
        return 0

    # Simulate publishing
    spinner = Spinner("Publishing")
    for _ in range(10):
        spinner.render()
        time.sleep(0.1)
    spinner.done("Published successfully")

    return 0


def cmd_status(ctx: CLIContext) -> int:
    """Show system status."""
    ctx.info("System Status")

    status_items = [
        ("Database", "OK", Colors.GREEN),
        ("LinkedIn API", "Connected", Colors.GREEN),
        ("Rate Limiter", "5 requests/min available", Colors.YELLOW),
        ("Scheduler", "Next run in 2h 15m", Colors.BLUE),
    ]

    for name, status, color in status_items:
        status_colored = colorize(status, color) if ctx.color else status
        print(f"  {name}: {status_colored}")

    return 0


def cmd_config(ctx: CLIContext) -> int:
    """Show or edit configuration."""
    key = ctx.args.get("key")
    value = ctx.args.get("value")

    if key and value:
        if ctx.dry_run:
            ctx.warning(f"Would set {key} = {value} (dry run)")
        else:
            ctx.success(f"Set {key} = {value}")
    elif key:
        ctx.info(f"{key} = <value>")
    else:
        ctx.info("Configuration:")
        ctx.echo("  (use 'config <key>' to view a specific setting)")
        ctx.echo("  (use 'config <key> <value>' to set a setting)")

    return 0


def create_default_cli() -> CLI:
    """Create CLI with default commands."""
    cli = CLI(name="smp", version="1.0.0")

    # Search command
    cli.add_command(
        Command(
            name="search",
            help="Search for news stories",
            handler=cmd_search,
            options=[
                Option("count", "c", "Number of stories to find", int, 5),
                Option("topic", "t", "Filter by topic", str),
            ],
        )
    )

    # Publish command
    cli.add_command(
        Command(
            name="publish",
            help="Publish a story to LinkedIn",
            handler=cmd_publish,
            options=[
                Option("force", "f", "Skip validation", is_flag=True),
            ],
            arguments=[
                Argument("story_id", "ID of story to publish", int),
            ],
        )
    )

    # Status command
    cli.add_command(
        Command(
            name="status",
            help="Show system status",
            handler=cmd_status,
        )
    )

    # Config command
    cli.add_command(
        Command(
            name="config",
            help="View or edit configuration",
            handler=cmd_config,
            arguments=[
                Argument("key", "Configuration key", str, False),
                Argument("value", "New value to set", str, False),
            ],
        )
    )

    return cli


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> "TestSuite":
    """Create unit tests for this module."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_framework import TestSuite

    suite = TestSuite("CLI")

    def test_verbosity_levels():
        assert Verbosity.QUIET.value == 0
        assert Verbosity.NORMAL.value == 1
        assert Verbosity.VERBOSE.value == 2
        assert Verbosity.DEBUG.value == 3

    def test_colorize_disabled():
        # Test without colors
        result = colorize("test", Colors.RED)
        assert "test" in result

    def test_progress_bar_init():
        bar = ProgressBar(total=100)
        assert bar.current == 0
        assert bar.total == 100

    def test_progress_bar_update():
        bar = ProgressBar(total=10)
        bar.update(5)
        assert bar.current == 5
        bar.update(3)
        assert bar.current == 8

    def test_progress_bar_set():
        bar = ProgressBar(total=100)
        bar.set(50)
        assert bar.current == 50
        bar.set(200)  # Should cap at total
        assert bar.current == 100

    def test_spinner_step():
        spinner = Spinner("Testing")
        frame1 = spinner.step()
        frame2 = spinner.step()
        assert "Testing" in frame1
        assert "Testing" in frame2
        assert frame1 != frame2

    def test_option_matches():
        opt = Option("verbose", "v", "Verbose output")
        assert opt.matches("--verbose")
        assert opt.matches("-v")
        assert not opt.matches("--quiet")

    def test_argument_defaults():
        arg = Argument("file", "File to process", str, False, "default.txt")
        assert arg.default == "default.txt"
        assert not arg.required

    def test_command_help():
        cmd = Command(
            name="test",
            help="Test command",
            handler=lambda ctx: 0,
            options=[Option("force", "f", "Force it", is_flag=True)],
        )
        help_text = cmd.format_help()
        assert "test" in help_text
        assert "force" in help_text

    def test_cli_context_verbosity():
        ctx = CLIContext(verbose=Verbosity.QUIET)
        # Should not raise
        ctx.echo("test", Verbosity.QUIET)

    def test_cli_add_command():
        cli = CLI()
        cmd = Command("test", "Test", lambda ctx: 0)
        cli.add_command(cmd)
        assert "test" in cli.commands

    def test_cli_completions():
        cli = create_default_cli()
        comps = cli.get_completions("s")
        assert "search" in comps
        assert "status" in comps

    def test_argument_parser():
        cmd = Command(
            name="test",
            help="Test",
            handler=lambda ctx: 0,
            options=[Option("count", "c", "Count", int, 5)],
        )
        parser = ArgumentParser(cmd)
        parsed, _ = parser.parse(["--count", "10"])
        assert parsed["count"] == 10

    def test_cli_help_flag():
        cli = create_default_cli()
        result = cli.run(["--help"])
        assert result == 0

    def test_cli_version_flag():
        cli = create_default_cli()
        result = cli.run(["--version"])
        assert result == 0

    suite.add_test("Verbosity levels", test_verbosity_levels)
    suite.add_test("Colorize disabled", test_colorize_disabled)
    suite.add_test("Progress bar init", test_progress_bar_init)
    suite.add_test("Progress bar update", test_progress_bar_update)
    suite.add_test("Progress bar set", test_progress_bar_set)
    suite.add_test("Spinner step", test_spinner_step)
    suite.add_test("Option matches", test_option_matches)
    suite.add_test("Argument defaults", test_argument_defaults)
    suite.add_test("Command help", test_command_help)
    suite.add_test("CLI context verbosity", test_cli_context_verbosity)
    suite.add_test("CLI add command", test_cli_add_command)
    suite.add_test("CLI completions", test_cli_completions)
    suite.add_test("Argument parser", test_argument_parser)
    suite.add_test("CLI help flag", test_cli_help_flag)
    suite.add_test("CLI version flag", test_cli_version_flag)

    return suite


if __name__ == "__main__":
    cli = create_default_cli()
    sys.exit(cli.run())
