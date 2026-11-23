"""
Enforce the code conventions of CONTRIBUTING.md: exports, constructor arguments, naming, docstrings
and folder READMEs. Each check returns a list of problems so failures name every offender
"""

import ast
import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from types import ModuleType

import pytest
from torch import nn

import torch_omni_tool
from torch_omni_tool.blocks.ViT import PositionEmbeddingSine

ROOT = Path(torch_omni_tool.__file__).parent
SCRIPTS = ROOT.parent / "scripts"

# exported modules without parameters or buffers do not need device / dtype
NO_PARAMETERS: tuple[type[nn.Module], ...] = (PositionEmbeddingSine,)
FORBIDDEN_ARGS = {
    "nf",
    "gc",
    "dim",
    "channels",
    "chans",
    "in_chan",
    "out_chan",
    "in_chans",
    "out_chans",
    "n_in",
    "n_out",
    "in_ch",
    "out_ch",
    "reduce_dim",
    "num_channels",
}
SECTION = re.compile(r"^[A-Z][A-Za-z ]*:$")
ARG_ITEM = re.compile(r"^(\*{0,2}\w+) \((.+?)\): \S")
RETURN_ITEM = re.compile(r"^(.+?): \S")


def _all_packages() -> list[ModuleType]:
    """
    Import every package of torch_omni_tool, the root included
    Returns:
        list[ModuleType]: Packages sorted by name
    """
    packages = [torch_omni_tool]
    for info in pkgutil.walk_packages(torch_omni_tool.__path__, "torch_omni_tool."):
        module = importlib.import_module(info.name)
        if info.ispkg:
            packages.append(module)
    return sorted(packages, key=lambda p: p.__name__)


PACKAGES = _all_packages()


def _exported_modules_classes() -> list[type[nn.Module]]:
    """
    Collect every nn.Module subclass exported by a package
    Returns:
        list[type[nn.Module]]: Exported module classes sorted by name
    """
    found: dict[str, type[nn.Module]] = {}
    for package in PACKAGES:
        for name in getattr(package, "__all__", []):
            obj = getattr(package, name, None)
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                found[f"{obj.__module__}.{obj.__name__}"] = obj
    return [found[k] for k in sorted(found)]


def _source_files() -> list[Path]:
    """
    List the Python files whose docstrings and names are checked
    Returns:
        list[Path]: Files of the package and of scripts/
    """
    return sorted([*ROOT.rglob("*.py"), *SCRIPTS.rglob("*.py")])


def _readme_folders() -> list[Path]:
    """
    List the folders that must have a README with a Source section
    Returns:
        list[Path]: Block and augmentation folders, plus activations and normalizations
    """
    folders = [ROOT / "activations", ROOT / "normalizations"]
    for parent in (ROOT / "blocks", ROOT / "datasets"):
        folders += [p for p in parent.iterdir() if (p / "__init__.py").exists()]
    return sorted(folders)


# checks


def _export_problems(package: ModuleType) -> list[str]:
    """
    Check that a package only exports what its own folder holds, under the real names
    Args:
        package (ModuleType): Package to check
    Returns:
        list[str]: Problems found, empty when the package follows the rules
    """
    pkg = package.__name__
    names = getattr(package, "__all__", None)
    if names is None:
        return [f"{pkg} has no __all__"]
    problems = []
    for name in names:
        obj = getattr(package, name, None)
        if obj is None:
            problems.append(f"{pkg}.__all__ lists {name!r} but it is not defined")
        elif inspect.ismodule(obj):
            if obj.__name__ != f"{pkg}.{name}":
                problems.append(f"{pkg} exports module {obj.__name__} as {name!r}")
        else:
            real = getattr(obj, "__name__", None)
            if real != name:
                problems.append(f"{pkg} exports {real!r} under the alias {name!r}")
            defined_in = getattr(obj, "__module__", "")
            if defined_in.rsplit(".", 1)[0] != pkg:
                problems.append(
                    f"{pkg} exports {name!r} from {defined_in}, export it from the folder that "
                    "holds its file instead"
                )
    for info in pkgutil.iter_modules(getattr(package, "__path__", [])):
        attr = getattr(package, info.name, None)
        if attr is not None and not inspect.ismodule(attr):
            problems.append(f"{pkg}.{info.name} is {attr!r}, which shadows the file {info.name}.py")
    return problems


def _signature_problems(cls: type[nn.Module]) -> list[str]:
    """
    Check that a module constructor follows torch's argument names and order
    Args:
        cls (type[nn.Module]): Module class to check
    Returns:
        list[str]: Problems found, empty when the constructor follows the rules
    """
    params = list(inspect.signature(cls.__init__).parameters.values())[1:]
    names = [p.name for p in params]
    where = f"{cls.__module__}.{cls.__name__}({', '.join(names)})"
    problems = []
    for name in names:
        if name in FORBIDDEN_ARGS or ("chan" in name and not name.endswith("_channels")):
            problems.append(
                f"{where}: rename {name!r} (see CONTRIBUTING.md, constructor arguments)"
            )
    if "in_channels" in names or "out_channels" in names:
        if names[:2] != ["in_channels", "out_channels"]:
            problems.append(f"{where}: start with in_channels, out_channels")
    elif "num_features" in names and names[0] != "num_features":
        problems.append(f"{where}: num_features must be the first argument")
    if not issubclass(cls, NO_PARAMETERS):
        tail = params[-2:]
        if [p.name for p in tail] != ["device", "dtype"] or any(
            p.default is not None for p in tail
        ):
            problems.append(f"{where}: end with device=None, dtype=None")
    return problems


def _name_problems(path: Path) -> list[str]:
    """
    Check the dimension suffixes of a file name and of the classes it defines
    Args:
        path (Path): Python file to check
    Returns:
        list[str]: Problems found, empty when the names follow the rules
    """
    problems = []
    if re.search(r"_[123]D\.py$", path.name):
        problems.append(f"{path.name}: use a lowercase dimension suffix (_1d.py, _2d.py, _3d.py)")
    file_dim = re.search(r"_([123])d\.py$", path.name)
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ClassDef):
            continue
        if re.search(r"[123]D$", node.name):
            problems.append(f"{path.name}: class {node.name} needs a lowercase 1d/2d/3d suffix")
        class_dim = re.search(r"([123])d$", node.name)
        if file_dim and class_dim and class_dim.group(1) != file_dim.group(1):
            problems.append(f"{path.name}: class {node.name} does not match the file dimension")
    return problems


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """
    Tell whether a function is an @overload stub
    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function node
    Returns:
        bool: True for overload stubs
    """
    return any(ast.unparse(d).split(".")[-1] == "overload" for d in node.decorator_list)


def _parse_sections(
    lines: list[str], base: int, where: str, problems: list[str]
) -> dict[str, list[str]]:
    """
    Split the inner lines of a docstring into sections, checking the indentation
    Args:
        lines (list[str]): Docstring lines between the quotes
        base (int): Indentation of the docstring
        where (str): Location used in the problem messages
        problems (list[str]): Collects indentation problems
    Returns:
        dict[str, list[str]]: Items of every section keyed by its name, continuation lines joined
    """
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        indent = len(line) - len(line.lstrip())
        text = line.strip()
        if indent == base and SECTION.match(text):
            current = sections.setdefault(text[:-1], [])
        elif current is None:
            continue
        elif indent == base + 4:
            current.append(text)
        elif indent == base + 8 and current:
            current[-1] += " " + text
        else:
            problems.append(f"{where}: section line indented by {indent - base}, use 4 or 8")
    return sections


def _docstring_problems(source: str, filename: str) -> list[str]:
    """
    Check the layout of every docstring in a source file and that Args / Returns match
    Args:
        source (str): Python source code
        filename (str): Name used in the problem messages
    Returns:
        list[str]: Problems found, empty when every docstring follows the rules
    """
    raw_lines = source.splitlines()
    problems: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = getattr(node, "name", "<module>")
        where = f"{filename}:{getattr(node, 'lineno', 1)} {name}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_overload(node):
            continue
        first = node.body[0] if node.body else None
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            problems.append(f"{where}: missing docstring")
            continue
        raw = raw_lines[first.lineno - 1 : first.end_lineno]
        if raw[0].strip() != '"""' or raw[-1].strip() != '"""':
            problems.append(
                f'{where}: the text must start on the line after """ and end before """'
            )
            continue
        inner = raw[1:-1]
        text = "\n".join(inner)
        if "::" in text or ">>>" in text:
            continue  # code examples keep their layout
        if any(not line.strip() for line in inner):
            problems.append(f"{where}: blank line inside the docstring")
        sections = _parse_sections(inner, first.col_offset, where, problems)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg:
            args.append(node.args.vararg)
        if node.args.kwarg:
            args.append(node.args.kwarg)
        if args and args[0].arg in ("self", "cls"):
            args = args[1:]
        expected = [(a.arg, ast.unparse(a.annotation) if a.annotation else "") for a in args]
        documented = []
        for item in sections.get("Args", []):
            match = ARG_ITEM.match(item)
            if match is None:
                problems.append(f"{where}: Args item must be 'name (type): description': {item!r}")
            else:
                documented.append((match.group(1).lstrip("*"), match.group(2)))
        if documented != expected:
            problems.append(f"{where}: Args documents {documented}, signature has {expected}")
        returns = ast.unparse(node.returns) if node.returns else "None"
        items = sections.get("Returns", [])
        if returns == "None" and items:
            problems.append(f"{where}: Returns section on a function returning None")
        elif returns != "None":
            match = RETURN_ITEM.match(items[0]) if len(items) == 1 else None
            if match is None or match.group(1) != returns:
                problems.append(f"{where}: Returns must be one item '{returns}: description'")
    return problems


def _readme_problems(folder: Path) -> list[str]:
    """
    Check that a folder README cites its source
    Args:
        folder (Path): Block or augmentation folder
    Returns:
        list[str]: Problems found, empty when the README has a Source section linking the paper
    """
    readme = folder / "README.md"
    if not readme.exists():
        return [f"{folder.name}: missing README.md"]
    text = readme.read_text()
    if "\n## Source\n" not in text:
        return [f"{folder.name}/README.md: missing a '## Source' section"]
    source = text.split("\n## Source\n", 1)[1].split("\n## ", 1)[0]
    if not re.search(r"\]\(https://(arxiv\.org|doi\.org)/", source):
        return [f"{folder.name}/README.md: the Source section must link the paper (arXiv or DOI)"]
    return []


def _test_style_problems(source: str, filename: str) -> list[str]:
    """
    Check that a test module uses plain test functions and private, documented helpers
    Args:
        source (str): Python source code of the test module
        filename (str): Name used in the problem messages
    Returns:
        list[str]: Problems found, empty when the module follows the rules
    """
    problems = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            problems.append(f"{filename}: test class {node.name}, use plain test functions")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not (node.name.startswith("test_") or node.name.startswith("_")):
                problems.append(f"{filename}: helper {node.name} must start with an underscore")
            elif node.name.startswith("_") and ast.get_docstring(node) is None:
                problems.append(f"{filename}: helper {node.name} needs a docstring")
    return problems


# checks over the package


@pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.__name__)
def test_exports_mirror_the_tree(package: ModuleType) -> None:
    assert _export_problems(package) == []


@pytest.mark.parametrize("package", PACKAGES[1:], ids=lambda p: p.__name__)
def test_every_subpackage_is_exported_by_its_parent(package: ModuleType) -> None:
    parent_name, name = package.__name__.rsplit(".", 1)
    parent = importlib.import_module(parent_name)
    assert name in getattr(parent, "__all__", []), f"add {name!r} to {parent_name}.__all__"


@pytest.mark.parametrize("cls", _exported_modules_classes(), ids=lambda c: c.__name__)
def test_constructor_arguments_follow_torch(cls: type[nn.Module]) -> None:
    assert _signature_problems(cls) == []


@pytest.mark.parametrize("cls", NO_PARAMETERS, ids=lambda c: c.__name__)
def test_no_parameters_allowlist_is_accurate(cls: type[nn.Module]) -> None:
    module = cls()
    assert not list(module.parameters()) and not list(module.buffers())


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(ROOT.parent)))
def test_dimension_suffixes(path: Path) -> None:
    assert _name_problems(path) == []


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(ROOT.parent)))
def test_docstrings(path: Path) -> None:
    assert _docstring_problems(path.read_text(), path.name) == []


@pytest.mark.parametrize("folder", _readme_folders(), ids=lambda p: p.name)
def test_folder_readme_cites_source(folder: Path) -> None:
    assert _readme_problems(folder) == []


@pytest.mark.parametrize(
    "path", sorted((ROOT.parent / "tests").glob("test_*.py")), ids=lambda p: p.name
)
def test_test_modules_use_functions_and_private_helpers(path: Path) -> None:
    assert _test_style_problems(path.read_text(), path.name) == []


# the checks catch broken code


GOOD = '''
def f(x: int, scale: float = 1.0) -> int:
    """
    Short description
    Args:
        x (int): Description of x
        scale (float): Description that is long enough to continue on the next line with
            an eight space indent
    Returns:
        int: Description of the result
    """
    return x
'''


def test_docstring_check_accepts_the_documented_style() -> None:
    assert _docstring_problems('"""\nModule\n"""\n' + GOOD, "good.py") == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ('    """\n    Short description', '    """Short description', "start on the line after"),
        ("    Short description\n", "    Short description\n\n", "blank line"),
        ("x (int): Description", "x (float): Description", "Args documents"),
        (
            "        scale (float): Description that is long enough to continue on the next line"
            " with\n            an eight space indent\n",
            "",
            "Args documents",
        ),
        ("        x (int)", "      x (int)", "indented by 2"),
        ("    Returns:\n        int: Description of the result\n", "", "Returns must be"),
        ("x (int): Description", "x: Description", "'name (type): description'"),
    ],
    ids=["same-line", "blank-line", "wrong-type", "missing-arg", "indent", "no-returns", "no-type"],
)
def test_docstring_check_rejects_broken_docstrings(old: str, new: str, expected: str) -> None:
    broken = GOOD.replace(old, new)
    assert broken != GOOD
    problems = _docstring_problems('"""\nModule\n"""\n' + broken, "bad.py")
    assert any(expected in p for p in problems), problems


def test_docstring_check_requires_docstrings() -> None:
    problems = _docstring_problems('"""\nModule\n"""\ndef f() -> None:\n    pass\n', "bad.py")
    assert any("missing docstring" in p for p in problems)


class _Good(nn.Module):
    """
    Constructor that follows the rules
    """

    def __init__(
        self, in_channels: int, out_channels: int, device: None = None, dtype: None = None
    ) -> None:
        """
        Build it
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            device (None): Device
            dtype (None): Data type
        """
        super().__init__()


class _Abbreviated(nn.Module):
    """
    Constructor with an abbreviation and no factory arguments
    """

    def __init__(self, nf: int) -> None:
        """
        Build it
        Args:
            nf (int): Channels
        """
        super().__init__()


class _Swapped(nn.Module):
    """
    Constructor with the channel arguments after another argument
    """

    def __init__(
        self,
        num_tasks: int,
        in_channels: int,
        out_channels: int,
        device: None = None,
        dtype: None = None,
    ) -> None:
        """
        Build it
        Args:
            num_tasks (int): Tasks
            in_channels (int): Input channels
            out_channels (int): Output channels
            device (None): Device
            dtype (None): Data type
        """
        super().__init__()


def test_signature_check() -> None:
    assert _signature_problems(_Good) == []
    problems = _signature_problems(_Abbreviated)
    assert any("rename 'nf'" in p for p in problems)
    assert any("device=None, dtype=None" in p for p in problems)
    assert any("start with in_channels" in p for p in _signature_problems(_Swapped))


def test_export_check(tmp_path: Path) -> None:
    (tmp_path / "Thing.py").write_text("")
    fake = ModuleType("fake")
    fake.__path__ = [str(tmp_path)]

    class Other:
        """
        Exported under an alias from a nested module
        """

    Other.__module__ = "fake.nested.module"
    fake.Thing = Other  # type: ignore[attr-defined]
    fake.__all__ = ["Thing"]  # type: ignore[attr-defined]
    problems = _export_problems(fake)
    assert any("alias" in p for p in problems)
    assert any("export it from the folder" in p for p in problems)
    assert any("shadows the file Thing.py" in p for p in problems)


def test_name_check(tmp_path: Path) -> None:
    path = tmp_path / "Block_2D.py"
    path.write_text("class Block3D:\n    pass\n")
    problems = _name_problems(path)
    assert any("lowercase dimension suffix" in p for p in problems)
    assert any("lowercase 1d/2d/3d suffix" in p for p in problems)
    mismatched = tmp_path / "Block_2d.py"
    mismatched.write_text("class Block3d:\n    pass\n")
    assert any("does not match" in p for p in _name_problems(mismatched))


def test_readme_check(tmp_path: Path) -> None:
    folder = tmp_path / "Block"
    folder.mkdir()
    assert _readme_problems(folder) == ["Block: missing README.md"]
    (folder / "README.md").write_text("# Block\n\n## Usage\n")
    assert "missing a '## Source' section" in _readme_problems(folder)[0]
    (folder / "README.md").write_text(
        "# Block\n\n## Source\n\n- code on [GitHub](https://github.com/x/y)\n"
    )
    assert "must link the paper" in _readme_problems(folder)[0]


def test_test_style_check() -> None:
    source = "class TestThing:\n    pass\n\n\ndef helper() -> None:\n    pass\n"
    problems = _test_style_problems(source, "test_bad.py")
    assert any("test class TestThing" in p for p in problems)
    assert any("helper helper" in p for p in problems)
    undocumented = _test_style_problems("def _helper() -> None:\n    pass\n", "test_bad.py")
    assert any("needs a docstring" in p for p in undocumented)
    assert _test_style_problems("def test_ok() -> None:\n    pass\n", "test_ok.py") == []
