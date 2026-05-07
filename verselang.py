import sys
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class VerseLangError(Exception):
    """Raised for VerseLang syntax or runtime errors."""


class ReturnSignal(Exception):
    def __init__(self, value: Any):
        super().__init__(str(value))
        self.value = value


@dataclass
class FunctionDef:
    name: str
    params: List[str]
    body: List[str]
    home_book: str
    home_place: str


COMMANDMENT_DIGITS = {
    "nogods": 0,
    "noidols": 1,
    "noname": 2,
    "sabbath": 3,
    "honor": 4,
    "nomurder": 5,
    "noadultery": 6,
    "nosteal": 7,
    "nolies": 8,
    "nocovet": 9,
}

PLACE_FLAVORS = {"plain", "sweet", "spicy", "sour", "bitter", "divine", "umami"}


GUIDE = """
VERSELANG GUIDE
===============

VerseLang is a scripture-inspired esoteric programming language.

Core ideas:
- Programs are divided into books.
- Books contain places.
- Places have flavors.
- Variables must be full quoted verse names.
- Assignment uses taste.
- Output uses write.
- Conditionals use judges.
- Math uses word operators.

Important:
- Expressions are evaluated left-to-right.
- Do not split let statements across multiple lines.
"""


class VerseLangInterpreter:
    def __init__(self) -> None:
        self.books: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.book_places: Dict[str, List[str]] = {}
        self.place_flavors: Dict[Tuple[str, str], str] = {}
        self.mailboxes: Dict[Tuple[str, str], List[Any]] = {}
        self.functions: Dict[str, FunctionDef] = {}
        self.memo: Dict[Tuple[str, str, str, Tuple[Any, ...]], Any] = {}
        self.current_book: Optional[str] = None
        self.current_place: Optional[str] = None
        self.current_chapter: int = 0
        self.current_verse: int = 0

    def run(self, source: str) -> None:
        self._execute_block(self._preprocess(source))

    def _preprocess(self, source: str) -> List[str]:
        lines = []
        for raw_line in source.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
        return lines

    def _ensure_book(self, book: str) -> None:
        if book not in self.books:
            self.books[book] = {}
            self.book_places[book] = []

    def _ensure_place(self, book: str, place: str, flavor: str = "plain") -> None:
        self._ensure_book(book)
        if place not in self.books[book]:
            self.books[book][place] = {}
            self.book_places[book].append(place)
        self.place_flavors.setdefault((book, place), flavor)
        self.mailboxes.setdefault((book, place), [])

    def _require_location(self) -> Tuple[str, str]:
        if self.current_book is None or self.current_place is None:
            raise VerseLangError("No active book/place. Use book + go <place> first.")
        return self.current_book, self.current_place

    def _set_location(self, book: str, place: str) -> None:
        self._ensure_place(book, place)
        self.current_book = book
        self.current_place = place
        self.current_chapter = self.book_places[book].index(place) + 1
        self.current_verse = 0

    def _current_env(self) -> Dict[str, Any]:
        book, place = self._require_location()
        return self.books[book][place]

    def _current_flavor(self) -> str:
        book, place = self._require_location()
        return self.place_flavors[(book, place)]

    def _execute_block(self, lines: List[str]) -> None:
        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith("book "):
                body, next_i, book_name = self._collect_book(lines, i)
                self._execute_book(book_name, body)
                i = next_i
                continue

            if line.startswith("place "):
                self._declare_place(line)
                i += 1
                continue

            if line.startswith("go "):
                if self.current_book is None:
                    raise VerseLangError("Cannot go to a place without an active book.")
                self._set_location(self.current_book, line[3:].strip())
                i += 1
                continue

            if line.startswith("let "):
                name, expr = self._parse_assignment(line[4:])
                self._current_env()[name] = self._eval_expr(expr)
                i += 1
                continue

            if line.startswith("write "):
                self._write_output(self._eval_expr(line[6:].strip()))
                i += 1
                continue

            if line.startswith("say ") and " to " in line:
                payload, destination = line[4:].split(" to ", 1)
                self._send_message(payload.strip(), destination.strip())
                i += 1
                continue

            if line.startswith("listen "):
                target = self._parse_quoted_name(line[7:].strip())
                book, place = self._require_location()
                mailbox = self.mailboxes[(book, place)]
                self._current_env()[target] = mailbox.pop(0) if mailbox else None
                i += 1
                continue

            if line.endswith(" judges"):
                true_block, false_block, next_i, condition = self._collect_judges(lines, i)
                if self._truthy(self._eval_expr(condition)):
                    self._execute_block(true_block)
                else:
                    self._execute_block(false_block)
                i = next_i
                continue

            if line.startswith("while "):
                body, next_i, condition = self._collect_while(lines, i)
                while self._truthy(self._eval_expr(condition)):
                    self._execute_block(body)
                i = next_i
                continue

            if line.startswith("func "):
                function_def, next_i = self._collect_function(lines, i)
                self.functions[function_def.name] = function_def
                i = next_i
                continue

            if line.startswith("return "):
                raise ReturnSignal(self._eval_expr(line[7:].strip()))

            if line in {"end", "else", "endbook"}:
                raise VerseLangError(f"Unexpected token: {line}")

            self._eval_expr(line)
            i += 1

    def _execute_book(self, name: str, body: List[str]) -> None:
        self._ensure_book(name)
        old_state = (self.current_book, self.current_place, self.current_chapter, self.current_verse)
        self.current_book = name
        self.current_place = None
        self.current_chapter = 0
        self.current_verse = 0
        self._execute_block(body)
        self.current_book, self.current_place, self.current_chapter, self.current_verse = old_state

    def _declare_place(self, line: str) -> None:
        parts = shlex.split(line)
        if len(parts) != 4 or parts[2] != "flavor":
            raise VerseLangError("Place syntax: place <name> flavor <flavor>")
        _, place_name, _, flavor = parts
        if flavor not in PLACE_FLAVORS:
            raise VerseLangError(f"Unknown flavor '{flavor}'")
        if self.current_book is None:
            raise VerseLangError("Place declarations must happen inside a book.")
        self._ensure_place(self.current_book, place_name, flavor)
        self.place_flavors[(self.current_book, place_name)] = flavor

    def _send_message(self, payload_expr: str, destination: str) -> None:
        if ":" not in destination:
            raise VerseLangError("say destination must be Book:Place")
        book_name, place_name = destination.split(":", 1)
        self._ensure_place(book_name, place_name)
        self.mailboxes[(book_name, place_name)].append(self._eval_expr(payload_expr))

    def _collect_book(self, lines: List[str], start: int) -> Tuple[List[str], int, str]:
        book_name = lines[start][5:].strip()
        body = []
        depth = 0
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.startswith("book "):
                depth += 1
                body.append(line)
            elif line == "endbook":
                if depth == 0:
                    return body, i + 1, book_name
                depth -= 1
                body.append(line)
            else:
                body.append(line)
            i += 1
        raise VerseLangError("Unclosed book block.")

    def _collect_judges(self, lines: List[str], start: int) -> Tuple[List[str], List[str], int, str]:
        condition = lines[start][:-7].strip()
        true_block = []
        false_block = []
        target = true_block
        depth = 0
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.endswith(" judges") or line.startswith(("while ", "func ", "book ")):
                depth += 1
                target.append(line)
            elif line == "end":
                if depth == 0:
                    return true_block, false_block, i + 1, condition
                depth -= 1
                target.append(line)
            elif line == "else" and depth == 0:
                target = false_block
            else:
                target.append(line)
            i += 1
        raise VerseLangError("Unclosed judges block.")

    def _collect_while(self, lines: List[str], start: int) -> Tuple[List[str], int, str]:
        condition = lines[start][6:].strip()
        body = []
        depth = 0
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.endswith(" judges") or line.startswith(("while ", "func ", "book ")):
                depth += 1
                body.append(line)
            elif line == "end":
                if depth == 0:
                    return body, i + 1, condition
                depth -= 1
                body.append(line)
            else:
                body.append(line)
            i += 1
        raise VerseLangError("Unclosed while block.")

    def _collect_function(self, lines: List[str], start: int) -> Tuple[FunctionDef, int]:
        parts = shlex.split(lines[start])
        if len(parts) < 2:
            raise VerseLangError("Function syntax: func <name> <args...>")
        _, name, *params = parts
        home_book, home_place = self._require_location()
        body = []
        depth = 0
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.endswith(" judges") or line.startswith(("while ", "func ", "book ")):
                depth += 1
                body.append(line)
            elif line == "end":
                if depth == 0:
                    return FunctionDef(name, params, body, home_book, home_place), i + 1
                depth -= 1
                body.append(line)
            else:
                body.append(line)
            i += 1
        raise VerseLangError(f"Unclosed function '{name}'.")

    def _parse_assignment(self, text: str) -> Tuple[str, str]:
        if " taste " not in text:
            raise VerseLangError(
                'Assignment syntax: let "full verse name" taste <expr>. '
                "Make sure the whole let statement is on one line."
            )
        name_part, expr = text.split(" taste ", 1)
        return self._parse_quoted_name(name_part.strip()), expr.strip()

    def _parse_quoted_name(self, text: str) -> str:
        parts = shlex.split(text)
        if len(parts) != 1:
            raise VerseLangError("Verse variable names must be one quoted string.")
        return parts[0]

    def _eval_expr(self, expr: str) -> Any:
        expr = expr.strip()
        if not expr:
            return None
        if expr.startswith("call "):
            return self._eval_call(expr)
        return self._eval_tokens(shlex.split(expr))

    def _eval_call(self, expr: str) -> Any:
        parts = shlex.split(expr)
        if len(parts) < 2:
            raise VerseLangError("call requires a function name.")
        _, function_name, *raw_args = parts
        args = [self._resolve_token(arg) for arg in raw_args]
        return self._call_function(function_name, args)

    def _call_function(self, name: str, args: List[Any]) -> Any:
        if name not in self.functions:
            raise VerseLangError(f"Undefined function '{name}'.")
        function_def = self.functions[name]
        if len(args) != len(function_def.params):
            raise VerseLangError(
                f"Function '{name}' expected {len(function_def.params)} args, got {len(args)}."
            )
        memo_key = (function_def.home_book, function_def.home_place, name, tuple(args))
        if self.place_flavors[(function_def.home_book, function_def.home_place)] == "umami":
            if memo_key in self.memo:
                return self.memo[memo_key]
        old_state = (self.current_book, self.current_place, self.current_chapter, self.current_verse)
        self._set_location(function_def.home_book, function_def.home_place)
        env = self._current_env()
        backup_env = dict(env)
        try:
            for param, arg in zip(function_def.params, args):
                env[param] = arg
            try:
                self._execute_block(function_def.body)
                result = None
            except ReturnSignal as signal:
                result = signal.value
        finally:
            self.books[function_def.home_book][function_def.home_place] = backup_env
            self.current_book, self.current_place, self.current_chapter, self.current_verse = old_state
        if self.place_flavors[(function_def.home_book, function_def.home_place)] == "umami":
            self.memo[memo_key] = result
        return result

    def _eval_tokens(self, tokens: List[str]) -> Any:
        if not tokens:
            return None
        value = self._resolve_token(tokens[0])
        i = 1
        while i < len(tokens):
            if i + 1 >= len(tokens):
                raise VerseLangError(f"Malformed expression: {' '.join(tokens)}")
            op = tokens[i]
            right = self._resolve_token(tokens[i + 1])
            value = self._apply_operator(value, op, right)
            i += 2
        return value

    def _resolve_token(self, token: str) -> Any:
        if token in COMMANDMENT_DIGITS:
            return COMMANDMENT_DIGITS[token]
        if "-" in token:
            parts = token.split("-")
            if all(part in COMMANDMENT_DIGITS for part in parts):
                return int("".join(str(COMMANDMENT_DIGITS[part]) for part in parts))
        if token == "here":
            return self.current_place
        if token == "book":
            return self.current_book
        if self.current_book is not None and self.current_place is not None:
            env = self._current_env()
            if token in env:
                return env[token]
        if token == "true":
            return True
        if token == "false":
            return False
        try:
            if "." in token:
                return float(token)
            return int(token)
        except ValueError:
            return token

    def _apply_operator(self, left: Any, op: str, right: Any) -> Any:
        if op == "sweet":
            if isinstance(left, str) or isinstance(right, str):
                return f"{left} {right}".strip()
            bonus = 1 if self._current_flavor() == "sweet" else 0
            return left + right + bonus
        if op == "sour":
            penalty = 1 if self._current_flavor() == "sour" else 0
            return left - right - penalty
        if op == "spicy":
            value = left * right
            return value * 2 if self._current_flavor() == "spicy" else value
        if op == "bitter":
            return left / right
        if op == "divine":
            return left % right
        if op == "is":
            return left == right
        if op == "not":
            return left != right
        if op == "greater":
            return left > right
        if op == "less":
            return left < right
        raise VerseLangError(f"Unknown operator '{op}'.")

    def _truthy(self, value: Any) -> bool:
        return bool(value)

    def _write_output(self, value: Any) -> None:
        if self.current_book is None:
            raise VerseLangError("Cannot write without an active book.")
        if self.current_chapter == 0:
            self.current_chapter = 1
        self.current_verse += 1
        print(f"{self.current_book} {self.current_chapter}:{self.current_verse} - {value}")


def main() -> None:
    interpreter = VerseLangInterpreter()

    if len(sys.argv) == 1:
        print(GUIDE)
        return

    if len(sys.argv) == 2 and sys.argv[1] == "--guide":
        print(GUIDE)
        return

    if len(sys.argv) != 2:
        print("Usage:")
        print("  python verselang.py --guide")
        print("  python verselang.py program.verse")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as file:
        source = file.read()
    interpreter.run(source)


if __name__ == "__main__":
    main()
