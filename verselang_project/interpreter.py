import sys
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class VerseLangError(Exception):
    """Raised for parsing or runtime errors in VerseLang."""


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


# Shifted commandment-decimal mapping.
# These are decimal digits, not standalone 1..10 values.
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


class VerseLangInterpreter:
    """
    VerseLang
    ---------
    A scripture-themed esoteric language.

    Main ideas:
    - Programs are organized into books.
    - Books contain places.
    - Places have flavors.
    - Variables must be full quoted verse names.
    - Assignment uses the word: taste
    - Output uses the word: write
    - Conditionals use the word: judges
    - Numbers can be written as commandment-digit words.

    Operators:
        sweet   -> addition
        sour    -> subtraction
        spicy   -> multiplication
        bitter  -> division
        divine  -> modulo
        is      -> equality
        not     -> inequality
        greater -> greater than
        less    -> less than

    Expressions are evaluated left-to-right.
    """

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

    # -----------------------------
    # Public API
    # -----------------------------
    def run(self, source: str) -> None:
        lines = self._preprocess(source)
        self._execute_block(lines)

    # -----------------------------
    # Preprocess
    # -----------------------------
    def _preprocess(self, source: str) -> List[str]:
        out: List[str] = []
        for raw in source.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
        return out

    # -----------------------------
    # World state helpers
    # -----------------------------
    def _ensure_book(self, book: str) -> None:
        if book not in self.books:
            self.books[book] = {}
            self.book_places[book] = []

    def _ensure_place(self, book: str, place: str, flavor: str = "plain") -> None:
        self._ensure_book(book)
        if place not in self.books[book]:
            self.books[book][place] = {}
            self.book_places[book].append(place)
        key = (book, place)
        if key not in self.place_flavors:
            self.place_flavors[key] = flavor
        if key not in self.mailboxes:
            self.mailboxes[key] = []

    def _require_location(self) -> Tuple[str, str]:
        if self.current_book is None or self.current_place is None:
            raise VerseLangError("No active book/place. Use enter <Book> and go <place> first.")
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

    # -----------------------------
    # Execution
    # -----------------------------
    def _execute_block(self, lines: List[str]) -> None:
        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith("book "):
                body, next_i, book_name = self._collect_book(lines, i)
                self._execute_book(book_name, body)
                i = next_i
                continue

            if line.startswith("enter "):
                book_name = line[6:].strip()
                if book_name not in self.books:
                    raise VerseLangError(f"Unknown book '{book_name}'")
                self.current_book = book_name
                self.current_place = None
                self.current_chapter = 0
                self.current_verse = 0
                i += 1
                continue

            if line.startswith("place "):
                self._declare_place(line)
                i += 1
                continue

            if line.startswith("go "):
                place_name = line[3:].strip()
                if self.current_book is None:
                    raise VerseLangError("Use enter <Book> before go <place>")
                self._set_location(self.current_book, place_name)
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
                true_block, false_block, next_i, cond = self._collect_judges(lines, i)
                if self._truthy(self._eval_expr(cond)):
                    self._execute_block(true_block)
                else:
                    self._execute_block(false_block)
                i = next_i
                continue

            if line.startswith("while "):
                body, next_i, cond = self._collect_while(lines, i)
                while self._truthy(self._eval_expr(cond)):
                    self._execute_block(body)
                i = next_i
                continue

            if line.startswith("func "):
                fn, next_i = self._collect_function(lines, i)
                self.functions[fn.name] = fn
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
        prev = (self.current_book, self.current_place, self.current_chapter, self.current_verse)
        self.current_book = name
        self.current_place = None
        self.current_chapter = 0
        self.current_verse = 0
        self._execute_block(body)
        self.current_book, self.current_place, self.current_chapter, self.current_verse = prev

    def _declare_place(self, line: str) -> None:
        parts = shlex.split(line)
        if len(parts) != 4 or parts[2] != "flavor":
            raise VerseLangError("Place syntax: place <name> flavor <plain|sweet|spicy|sour|bitter|divine|umami>")
        _, place_name, _, flavor = parts
        if self.current_book is None:
            raise VerseLangError("Place declarations must happen inside a book or after enter <Book>")
        if flavor not in PLACE_FLAVORS:
            raise VerseLangError(f"Unknown flavor '{flavor}'. Allowed: {sorted(PLACE_FLAVORS)}")
        self._ensure_place(self.current_book, place_name, flavor)
        self.place_flavors[(self.current_book, place_name)] = flavor

    def _send_message(self, payload_expr: str, destination: str) -> None:
        if ":" not in destination:
            raise VerseLangError("say destination must be Book:Place")
        book_name, place_name = destination.split(":", 1)
        self._ensure_place(book_name, place_name)
        self.mailboxes[(book_name, place_name)].append(self._eval_expr(payload_expr))

    # -----------------------------
    # Block collectors
    # -----------------------------
    def _collect_book(self, lines: List[str], start: int) -> Tuple[List[str], int, str]:
        name = lines[start][5:].strip()
        body: List[str] = []
        depth = 0
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.startswith("book "):
                depth += 1
                body.append(line)
            elif line == "endbook":
                if depth == 0:
                    return body, i + 1, name
                depth -= 1
                body.append(line)
            else:
                body.append(line)
            i += 1
        raise VerseLangError("Unclosed book block")

    def _collect_judges(self, lines: List[str], start: int) -> Tuple[List[str], List[str], int, str]:
        cond = lines[start][:-7].strip()
        truthy_block: List[str] = []
        falsy_block: List[str] = []
        target = truthy_block
        depth = 0
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.endswith(" judges") or line.startswith(("while ", "func ", "book ")):
                depth += 1
                target.append(line)
            elif line == "end":
                if depth == 0:
                    return truthy_block, falsy_block, i + 1, cond
                depth -= 1
                target.append(line)
            elif line == "else" and depth == 0:
                target = falsy_block
            else:
                target.append(line)
            i += 1
        raise VerseLangError("Unclosed judges block")

    def _collect_while(self, lines: List[str], start: int) -> Tuple[List[str], int, str]:
        cond = lines[start][6:].strip()
        body: List[str] = []
        depth = 0
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.endswith(" judges") or line.startswith(("while ", "func ", "book ")):
                depth += 1
                body.append(line)
            elif line == "end":
                if depth == 0:
                    return body, i + 1, cond
                depth -= 1
                body.append(line)
            else:
                body.append(line)
            i += 1
        raise VerseLangError("Unclosed while block")

    def _collect_function(self, lines: List[str], start: int) -> Tuple[FunctionDef, int]:
        parts = shlex.split(lines[start])
        if len(parts) < 2:
            raise VerseLangError("Invalid function definition")
        _, name, *params = parts
        home_book, home_place = self._require_location()

        body: List[str] = []
        depth = 0
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.endswith(" judges") or line.startswith(("while ", "func ", "book ")):
                depth += 1
                body.append(line)
            elif line == "end":
                if depth == 0:
                    fn = FunctionDef(name=name, params=params, body=body, home_book=home_book, home_place=home_place)
                    return fn, i + 1
                depth -= 1
                body.append(line)
            else:
                body.append(line)
            i += 1
        raise VerseLangError(f"Unclosed function '{name}'")

    # -----------------------------
    # Expression parsing
    # -----------------------------
    def _parse_assignment(self, text: str) -> Tuple[str, str]:
        marker = " taste "
        if marker not in text:
            raise VerseLangError('Assignment syntax: let "full verse name" taste <expr>')
        name_part, expr = text.split(marker, 1)
        return self._parse_quoted_name(name_part.strip()), expr.strip()

    def _parse_quoted_name(self, text: str) -> str:
        parts = shlex.split(text)
        if len(parts) != 1:
            raise VerseLangError("Verse variable names must be a single quoted string")
        return parts[0]

    def _eval_expr(self, expr: str) -> Any:
        expr = expr.strip()
        if not expr:
            return None
        if expr.startswith("call "):
            return self._eval_call(expr)
        tokens = shlex.split(expr)
        if not tokens:
            return None
        return self._eval_tokens(tokens)

    def _eval_call(self, expr: str) -> Any:
        parts = shlex.split(expr)
        if len(parts) < 2:
            raise VerseLangError("call requires a function name")
        _, fn_name, *arg_tokens = parts
        args = [self._resolve_token(tok) for tok in arg_tokens]
        return self._call_function(fn_name, args)

    def _call_function(self, name: str, args: List[Any]) -> Any:
        if name not in self.functions:
            raise VerseLangError(f"Undefined function '{name}'")
        fn = self.functions[name]
        if len(args) != len(fn.params):
            raise VerseLangError(f"Function '{name}' expected {len(fn.params)} args, got {len(args)}")

        memo_key = (fn.home_book, fn.home_place, name, tuple(args))
        if self.place_flavors[(fn.home_book, fn.home_place)] == "umami" and memo_key in self.memo:
            return self.memo[memo_key]

        prev = (self.current_book, self.current_place, self.current_chapter, self.current_verse)
        self._set_location(fn.home_book, fn.home_place)
        env = self._current_env()
        backup = dict(env)

        try:
            for param, arg in zip(fn.params, args):
                env[param] = arg
            try:
                self._execute_block(fn.body)
            except ReturnSignal as signal:
                result = signal.value
            else:
                result = None
        finally:
            self.books[fn.home_book][fn.home_place] = backup
            self.current_book, self.current_place, self.current_chapter, self.current_verse = prev

        if self.place_flavors[(fn.home_book, fn.home_place)] == "umami":
            self.memo[memo_key] = result
        return result

    def _eval_tokens(self, tokens: List[str]) -> Any:
        value = self._resolve_token(tokens[0])
        i = 1
        while i < len(tokens):
            if i + 1 >= len(tokens):
                raise VerseLangError(f"Malformed expression: {' '.join(tokens)}")
            op = tokens[i]
            rhs = self._resolve_token(tokens[i + 1])
            value = self._apply_operator(value, op, rhs)
            i += 2
        return value

    def _resolve_token(self, token: str) -> Any:
        if token in COMMANDMENT_DIGITS:
            return COMMANDMENT_DIGITS[token]

        if "-" in token:
            parts = token.split("-")
            if all(part in COMMANDMENT_DIGITS for part in parts):
                return int("".join(str(COMMANDMENT_DIGITS[p]) for p in parts))

        if token == "here":
            return self.current_place
        if token == "book":
            return self.current_book

        if ":" in token:
            parts = token.split(":")
            if len(parts) == 2:
                current_book, _ = self._require_location()
                place_name, name = parts
                self._ensure_place(current_book, place_name)
                if name == "flavor":
                    return self.place_flavors[(current_book, place_name)]
                return self.books[current_book][place_name].get(name)
            if len(parts) == 3:
                book_name, place_name, name = parts
                self._ensure_place(book_name, place_name)
                if name == "flavor":
                    return self.place_flavors[(book_name, place_name)]
                return self.books[book_name][place_name].get(name)
            raise VerseLangError("Cross reference must be Place:name or Book:Place:name")

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
            return self._op_sweet(left, right)
        if op == "sour":
            return self._op_sour(left, right)
        if op == "spicy":
            return self._op_spicy(left, right)
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
        raise VerseLangError(f"Unknown operator '{op}'")

    def _op_sweet(self, left: Any, right: Any) -> Any:
        if isinstance(left, str) or isinstance(right, str):
            return f"{left} {right}".strip()
        if self.current_book is not None and self.current_place is not None and self._current_flavor() == "sweet":
            return left + right + 1
        return left + right

    def _op_sour(self, left: Any, right: Any) -> Any:
        if self.current_book is not None and self.current_place is not None and self._current_flavor() == "sour":
            return left - right - 1
        return left - right

    def _op_spicy(self, left: Any, right: Any) -> Any:
        if self.current_book is not None and self.current_place is not None and self._current_flavor() == "spicy":
            return left * right * 2
        return left * right

    def _truthy(self, value: Any) -> bool:
        return bool(value)

    # -----------------------------
    # Output
    # -----------------------------
    def _write_output(self, value: Any) -> None:
        if self.current_book is None:
            raise VerseLangError("Cannot write without an active book")
        if self.current_chapter == 0:
            self.current_chapter = 1
        self.current_verse += 1
        print(f"{self.current_book} {self.current_chapter}:{self.current_verse} - {value}")


GUIDE = r'''
VERSELANG GUIDE
===============

What it is
----------
VerseLang is a scripture-themed esoteric language.
Programs are split into books. Books contain places. Places have flavors.
Variables must be full quoted names.

Core rules
----------
1. Variables must be full quoted strings.
2. Assignment uses the word taste.
3. Output uses the word write.
4. Math and comparison use words instead of symbols.
5. Expressions are evaluated left to right.
6. Functions always run in the place where they were defined.
7. Conditionals use judges after the condition.
8. Commandment words denote decimal digits.

Keywords
--------
book <Name>
endbook
enter <Name>
place <name> flavor <plain|sweet|spicy|sour|bitter|divine|umami>
go <place>
let "full verse name" taste <expr>
write <expr>
say <expr> to <Book>:<Place>
listen "full verse name"
<expr> judges
else
end
while <expr>
func <name> <arg1> <arg2> ...
return <expr>
call <name> <arg1> <arg2> ...

Commandment decimal digits
--------------------------
nogods      -> 0
noidols     -> 1
noname      -> 2
sabbath     -> 3
honor       -> 4
nomurder    -> 5
noadultery  -> 6
nosteal     -> 7
nolies      -> 8
nocovet     -> 9

You can combine them with hyphens:
    noidols-nogods   -> 10
    noidols-nomurder -> 15
    noname-noidols   -> 21

Operators
---------
sweet   -> addition, or string join with spaces
sour    -> subtraction
spicy   -> multiplication
bitter  -> division
divine  -> modulo
is      -> equality
not     -> inequality
greater -> greater than
less    -> less than
'''


HELLO_WORLD = r'''
book Genesis
    place garden flavor plain

    go garden
    write "hello world"
endbook

enter Genesis
go garden
'''


FIZZ_BUZZ = r'''
book Numbers
    place field flavor divine

    go field
    let "I am the way, the truth, and the life" taste noidols

    while "I am the way, the truth, and the life" less noname-noidols
        "I am the way, the truth, and the life" divine noidols-nomurder is nogods judges
            write "FizzBuzz"
        else
            "I am the way, the truth, and the life" divine sabbath is nogods judges
                write "Fizz"
            else
                "I am the way, the truth, and the life" divine nomurder is nogods judges
                    write "Buzz"
                else
                    write "I am the way, the truth, and the life"
                end
            end
        end

        let "I am the way, the truth, and the life" taste "I am the way, the truth, and the life" sweet noidols
    end
endbook

enter Numbers
go field
'''


SAMPLE_PROGRAM = r'''
book Genesis
    place garden flavor sweet
    place forge flavor spicy

    go garden
    let "I am the resurrection and the life" taste honor sweet noname
    write "garden holds" sweet "I am the resurrection and the life"
    say "fresh bread" to Genesis:forge

    nocovet greater honor judges
        write "the garden is abundant"
    else
        write "the garden is empty"
    end

    go forge
    listen "My peace I give unto you"
    write "the forge heard" sweet "My peace I give unto you"
    let "Fear not, for I am with you" taste honor spicy sabbath
    write "heat rises to" sweet "Fear not, for I am with you"
endbook
'''


def main() -> None:
    interpreter = VerseLangInterpreter()

    if len(sys.argv) == 1:
        print("Running bundled sample program...\n")
        interpreter.run(SAMPLE_PROGRAM)
        return

    if len(sys.argv) == 2 and sys.argv[1] == "--guide":
        print(GUIDE)
        print("\nHELLO WORLD\n===========\n")
        print(HELLO_WORLD)
        print("\nFIZZ BUZZ\n=========\n")
        print(FIZZ_BUZZ)
        print("\nSAMPLE PROGRAM\n==============\n")
        print(SAMPLE_PROGRAM)
        return

    if len(sys.argv) != 2:
        print("Usage:")
        print("  python verselang_interpreter.py")
        print("  python verselang_interpreter.py --guide")
        print("  python verselang_interpreter.py your_program.verse")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        program = handle.read()
    interpreter.run(program)


if __name__ == "__main__":
    main()
