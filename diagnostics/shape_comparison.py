"""Compare PR 2939's reported shape regression without uploading to CodSpeed."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tarfile
import time
import urllib.request


ROOT = Path.cwd()
WORK = Path(os.environ["RUNNER_TEMP"]) / "xtensor-shape-comparison"
OUT = ROOT / "shape-evidence"
PINS = {
    "base": ("xtensor-stack/xtensor", "8e14e7bf116b01130003873a98980043b06420b7"),
    "head": ("xtensor-stack/xtensor", "821ea730dde413457aefecc1f7977ceac452edee"),
    "xtl": ("xtensor-stack/xtl", "ef84e9f27020ad54961c99acd293d80d4b775dd3"),
    "xsimd": ("xtensor-stack/xsimd", "80c23624ce008d937da7e845e528e82ce0cbf4e0"),
    "benchmark": ("CodSpeedHQ/codspeed-cpp", "76985b6f02c49fe70393c50fcca6e9748f300caa"),
    "hooks": ("CodSpeedHQ/instrument-hooks", "1ec92c8c9db59db2d0adc37f47d003d3db4e1c64"),
    "gtest": ("google/googletest", "52eb8108c5bdec04579160ae17225d66034bd723"),
}
RUNNER_URL = (
    "https://github.com/CodSpeedHQ/codspeed/releases/download/v4.19.1/"
    "codspeed-runner-x86_64-unknown-linux-musl.tar.gz"
)
RUNNER_SHA256 = "9ea9a9c1e919f229b8b43c679c053eee513648cb08a826beb1a3ba7881a370ca"


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + "\n")


def run(label, args, cwd=ROOT, check=True):
    started = time.time()
    args = [str(x) for x in args]
    print(label, " ".join(args), flush=True)
    with (OUT / (label + ".log")).open("wb") as stream:
        result = subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    save(label + ".json", {"args": args, "cwd": str(cwd),
                           "exit_code": result.returncode,
                           "seconds": time.time() - started})
    if check and result.returncode:
        print((OUT / (label + ".log")).read_text(errors="replace")[-12000:])
        raise RuntimeError(f"{label}: exit {result.returncode}")
    return result.returncode


def download(url, target):
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as stream:
        shutil.copyfileobj(response, stream)


def main():
    WORK.mkdir()
    OUT.mkdir()
    save("pins.json", PINS)
    run("compiler", [os.environ["CXX"], "--version"])
    run("cpu", ["lscpu"])
    run("kernel", ["uname", "-a"])
    run("libraries", ["ldd", "--version"])
    sources = {}
    archives = {}
    for name, (repo, sha) in PINS.items():
        archive = WORK / (name + ".tar.gz")
        download(f"https://codeload.github.com/{repo}/tar.gz/{sha}", archive)
        archives[name] = digest(archive)
        with tarfile.open(archive) as tar:
            top = tar.getnames()[0].split("/")[0]
            assert top.endswith(sha), top
            tar.extractall(WORK / name, filter="data")
        sources[name] = WORK / name / top
    save("source-archive-sha256.json", archives)
    prefix = WORK / "prefix"
    for name in ["xtl", "xsimd"]:
        build = WORK / ("build-" + name)
        run(name + "-configure", ["cmake", "-S", sources[name], "-B", build,
                                  "-DBUILD_TESTS=OFF", "-DBUILD_TESTING=OFF",
                                  f"-DCMAKE_INSTALL_PREFIX={prefix}"])
        run(name + "-install", ["cmake", "--install", build])
    binaries = {}
    for name in ["base", "head"]:
        source, build = sources[name], WORK / ("build-" + name)
        run(name + "-configure", [
            "cmake", "-G", "Ninja", "-S", source, "-B", build,
            f"-DCMAKE_PREFIX_PATH={prefix}", "-DBUILD_BENCHMARK=ON",
            "-DXTENSOR_USE_XSIMD=ON", "-DCODSPEED_MODE=simulation",
            f"-DCODSPEED_ROOT_DIR={source}",
            "-DDOWNLOAD_GBENCHMARK=ON", "-DBENCHMARK_ENABLE_TESTING=OFF",
            "-DBENCHMARK_ENABLE_GTEST_TESTS=OFF", "-DBENCHMARK_ENABLE_WERROR=OFF",
            f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={sources['gtest']}",
            f"-DFETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK={sources['benchmark']}",
            f"-DFETCHCONTENT_SOURCE_DIR_INSTRUMENT_HOOKS_REPO={sources['hooks']}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ])
        run(name + "-build", ["cmake", "--build", build, "--target",
                              "benchmark_xtensor", "--parallel", "2"])
        binaries[name] = build / "benchmark/benchmark_xtensor"
        shutil.copy2(build / "compile_commands.json", OUT / (name + "-compile-commands.json"))
        obj = build / "benchmark/CMakeFiles/benchmark_xtensor.dir/benchmark_xshape.cpp.o"
        run(name + "-shape-disassembly", ["objdump", "-drC", obj])
        run(name + "-shape-sections", ["objdump", "-h", obj])
        run(name + "-binary-shape-symbols", ["nm", "-CS", binaries[name]])
        run(name + "-dependency-list", ["ninja", "-C", build, "-t", "deps",
                                        "benchmark/CMakeFiles/benchmark_xtensor.dir/benchmark_xshape.cpp.o"])
        run(name + "-benchmark-list", [binaries[name], "--benchmark_list_tests=true"])
        save(name + "-binary-sha256.json", {"binary": digest(binaries[name]), "shape_object": digest(obj)})
        shutil.copy2(obj, OUT / (name + "-shape.o"))
    runner_archive = WORK / "runner.tar.gz"
    download(RUNNER_URL, runner_archive)
    assert digest(runner_archive) == RUNNER_SHA256
    with tarfile.open(runner_archive) as tar:
        tar.extractall(WORK / "runner", filter="data")
    runners = list((WORK / "runner").rglob("codspeed")) + list((WORK / "runner").rglob("codspeed-runner"))
    runners = [x for x in runners if x.is_file() and os.access(x, os.X_OK)]
    assert len(runners) == 1, runners
    runner = runners[0]
    run("runner-version", [runner, "--version"])
    results = []
    for index, name in enumerate(["base", "head", "head", "base"]):
        profile = OUT / f"profile-{index}-{name}"
        profile.mkdir()
        label = f"simulation-{index}-{name}"
        code = run(label, [runner, "run", "--mode", "simulation", "--skip-upload",
                           "--profile-folder", profile, "--show-full-output", "--",
                           f"{binaries[name]} --benchmark_filter=xshape_access"], check=False)
        results.append({"index": index, "source": name, "exit_code": code})
    save("simulation-results.json", results)
    assert all(x["exit_code"] == 0 for x in results), results
    # Keep every compiled object unchanged and move only the shape object to
    # the front of the linker inputs. This controls its placement in .text.
    controls = {}
    for name in ["base", "head"]:
        build = WORK / ("build-" + name)
        run(name + "-link-commands", ["ninja", "-C", build, "-t", "commands",
                                      "benchmark_xtensor"])
        command = (OUT / (name + "-link-commands.log")).read_text().splitlines()[-1]
        tokens = shlex.split(command)
        assert tokens[:2] == [":", "&&"] and tokens[-2:] == ["&&", ":"], tokens
        tokens = tokens[2:-2]
        shape = "benchmark/CMakeFiles/benchmark_xtensor.dir/benchmark_xshape.cpp.o"
        assert tokens.count(shape) == 1
        tokens.remove(shape)
        first_object = next(i for i, value in enumerate(tokens) if value.endswith(".o"))
        tokens.insert(first_object, shape)
        control = build / "benchmark/benchmark_shape_first"
        assert tokens[tokens.index("-o") + 1] == "benchmark/benchmark_xtensor"
        tokens[tokens.index("-o") + 1] = str(control)
        run(name + "-link-shape-first", tokens, cwd=build)
        run(name + "-control-symbols", ["nm", "-CS", control])
        profile = OUT / ("profile-control-" + name)
        profile.mkdir()
        controls[name] = run(name + "-simulation-shape-first", [
            runner, "run", "--mode", "simulation", "--skip-upload",
            "--profile-folder", profile, "--show-full-output", "--",
            f"{control} --benchmark_filter=xshape_access",
        ], check=False)
    save("simulation-control-results.json", controls)
    assert all(code == 0 for code in controls.values()), controls


if __name__ == "__main__":
    main()
