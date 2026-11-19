#!/usr/bin/env python3
"""Saddle-search coverage harness.

The numeric protocol below is fixed before any timed round. Later edits
that change a tolerance, a cap, a spring, or a displacement after a
timed result exist are not part of this comparison.

Matched stop: the maximum per-atom force norm is below 0.01 eV/Angstrom.
That is ASE's fmax, eOn's max_atom metric, and rgsaddle's MAX_ATOM force
gate. Each code keeps its own optimizer.

Matched band: 7 and 11 images including both endpoints, uniform spring
5.0 eV/Angstrom^2, improved tangent, plain NEB projection, linear
initial path. eOn counts interior images, so its image count is 5 and 9.
Climbing image is off for plain NEB and on for CI-NEB. rgsaddle CI-NEB
sets the climbing-after flag with trigger factor 0 and trigger force
0.5, and turns min-mode refinement off, so the band is plain CI-NEB.
eOn climbing keeps its own default arm (immediate) and converged_only.

Matched min-mode start: 12 seeds, not a subset. Seed s for a system
is PCG64(20261118 + s + tag * 13). The tag is a stable mix of the
system name, so two systems that share a reactant do not draw the
same displacement. Each free Cartesian component is an independent
normal draw with standard deviation 0.1 Angstrom, which is eOn's
documented displace magnitude. The initial mode is that displacement
after it is normalized. Fixed components stay zero. The same displaced
geometry and the same mode are given to every code. Each code keeps
its own dimer spacing, rotation tolerance, and translator.

Cap: 1000 optimizer iterations, eOn's documented maximum. A run that
is still above the force gate is a recorded miss. Nothing is rerun
with a larger cap.

Reference, once, before the timed rounds, and not a timed cell:
rgsaddle CI-NEB, 15 images, MAX_ATOM gate 1e-4 eV/Angstrom, spring 5,
min-mode refinement off, at most 2000 steps. The point is kept only
when the library certificate says index 1. Timed success is that
certificate, plus the code stopping under the matched gate, plus the
saddle energy within 0.01 eV and the free-atom RMSD within 0.05
Angstrom of that point. The barrier is the saddle energy minus the
reactant energy and is compared at the same 0.01 eV.

RMSD uses free atoms only. A periodic pair is the Euclidean
minimum-image displacement from minimage. A free cluster is centered
and Kabsch-aligned. Identical species are matched by enumerating
permutations when that count is at most 5040.

One potential evaluation is one call of the rgpot evaluate method.
A final probe of the returned geometry is not part of the search
count or the search wall time. Wall time outside the potential is
the search wall time minus the time inside evaluate.

Rounds are interleaved. Round k runs every method of every system,
forward when k is even and reversed when k is odd. The same seed
is round k. Even rounds against odd rounds are the A/A split.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import zlib


PROTOCOL = {
    "fmax_ev_per_angstrom": 0.01,
    "force_gate": "max_per_atom_norm",
    "spring_k_ev_per_angstrom2": 5.0,
    "spring": "uniform",
    "tangent": "improved",
    "projection": "neb",
    "path": "linear",
    "images_including_endpoints": [7, 11],
    "pt_vacuum_angstrom": 25.0,
    "pt_pbc": "all",
    "lj7_product_order": "minimum total distance of identical atoms onto the reactant",
    "molecular": "cu4_h, hydrogen moving between two faces of a Cu4 tetrahedron, CuH2",
    "pt_cell_note": (
        "The Pt slab is periodic along every axis. The vacuum is 25 Angstrom "
        "so a 9.5 Angstrom Morse cutoff does not reach the next slab and the "
        "cell is taller than twice that cutoff. eOn Matter keeps one periodic "
        "flag, so the three codes share this cell instead of a mixed flag."
    ),
    "eon_interior_images": [5, 9],
    "max_iterations": 1000,
    "energy_tol_ev": 0.01,
    "barrier_tol_ev": 0.01,
    "rmsd_tol_angstrom": 0.05,
    "displacement_sigma_angstrom": 0.1,
    "displacement_generator": "PCG64",
    "displacement_seed_base": 20261118,
    "displacement_seed": (
        "PCG64(displacement_seed_base + round + tag * (n_seeds + 1)); "
        "tag is zlib.crc32 of the system name"
    ),
    "cluster_rmsd": (
        "center, Kabsch rotation, and enumeration of identical species "
        "when the permutation count is at most 5040"
    ),
    "n_seeds": 12,
    "reference_images": 15,
    "reference_fmax_ev_per_angstrom": 1.0e-4,
    "reference_max_iterations": 2000,
    "endpoint_fmax_ev_per_angstrom": 1.0e-3,
    "endpoint_max_iterations": 500,
    "rgsaddle_neb": {
        "method": "fire",
        "climbing": False,
        "min_mode_refinement": False,
        "max_move_angstrom": 0.2,
    },
    "rgsaddle_ci": {
        "method": "fire",
        "climbing": True,
        "ci_trigger_factor": 0.0,
        "ci_trigger_force": 0.5,
        "min_mode_refinement": False,
        "max_move_angstrom": 0.2,
    },
    "rgsaddle_minmode": {
        "translator": "fire",
        "dr_angstrom": 1.0e-3,
        "rotation_tol": 1.0e-4,
        "rotation_angle_tol_radian": 5.0 * math.pi / 180.0,
        "max_rotations": 20,
        "krylov_dim": 12,
        "max_move_angstrom": 0.2,
        "difference": "forward",
    },
    "eon_neb": {
        "optimizer": "lbfgs",
        "climbing": False,
        "convergence_metric": "max_atom",
    },
    "eon_ci": {
        "optimizer": "lbfgs",
        "climbing": True,
        "converged_only": True,
        "convergence_metric": "max_atom",
    },
    "eon_dimer": {"minmode": "dimer", "improved": True, "optimizer": "cg"},
    "eon_lanczos": {"minmode": "lanczos", "optimizer": "cg"},
    "ase_neb": {"optimizer": "bfgs", "method": "improvedtangent", "climb": False},
    "ase_ci": {"optimizer": "bfgs", "method": "improvedtangent", "climb": True},
    "ase_dimer": {
        "driver": "MinModeAtoms+MinModeTranslate",
        "eigenmode_method": "dimer",
    },
    "ase_lanczos": "not shipped; MinModeAtoms rejects any method other than dimer",
    "aa_split": "even rounds versus odd rounds",
    "order": "methods and systems forward on even rounds, reversed on odd rounds",
}

NEB_METHODS = (
    "rgsaddle-neb",
    "eon-neb",
    "ase-neb",
    "rgsaddle-ci",
    "eon-ci",
    "ase-ci",
)
MM_METHODS = (
    "rgsaddle-dimer",
    "eon-dimer",
    "ase-dimer",
    "rgsaddle-lanczos",
    "eon-lanczos",
)
SYSTEM_ORDER = ("pt_hop", "pt_exchange", "al_vacancy", "lj7", "cu4_h")

ABI_MAJOR = 1
ABI_MINOR = 42
FLAG_CI_AFTER = 1 << 2
FLAG_NO_MIN_MODE = 1 << 3
GATE_MAX_ATOM = 2
STALLED = -13


def parse_args(argv):
    parser = argparse.ArgumentParser(description="saddle-search coverage harness")
    parser.add_argument("--phase", required=True, choices=("setup", "measure"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--pythonpath", required=True, action="append")
    parser.add_argument("--libdir", required=True)
    parser.add_argument("--rgsaddle-lib", required=True)
    parser.add_argument("--cpu", required=True, type=int)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--rounds", required=True, type=int)
    parser.add_argument("--reference", required=False)
    parser.add_argument("--sha", required=True, action="append")
    args = parser.parse_args(argv)
    if args.cpu < 0 or args.threads < 1:
        parser.error("cpu and threads must be non-negative and positive")
    if args.rounds < 12:
        parser.error("rounds must be at least 12")
    if args.phase == "measure" and not args.reference:
        parser.error("measure requires --reference")
    shas = {}
    for item in args.sha:
        if "=" not in item:
            parser.error("sha entries are name=hex")
        name, value = item.split("=", 1)
        shas[name] = value
    for name in ("rgpot", "eon", "rgsaddle"):
        if name not in shas or len(shas[name]) < 8:
            parser.error("sha requires rgpot, eon, and rgsaddle")
    args.shas = shas
    return args


def pin(args):
    allowed = os.sched_getaffinity(0)
    if args.cpu not in allowed and args.cpu >= len(allowed):
        raise SystemExit("cpu is outside the processors this process may use")
    os.sched_setaffinity(0, {args.cpu})
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(args.threads)
    ctypes.CDLL(os.path.join(args.libdir, "librgpot.so"), mode=ctypes.RTLD_GLOBAL)
    preload_eonclib(args.pythonpath, args.libdir)
    for entry in reversed(args.pythonpath):
        if entry not in sys.path:
            sys.path.insert(0, entry)


EON_PROJECT_VERSION = "3.6.0"
PERMUTATION_CAP = 5040
RGSADDLE_BUILD = "release, C API"
RGPOT_BUILD = "scalar kernels, fast neighbour list on, Fortran pots on"

BUILD_PINS = (
    ("linkcell", "0.3.5 with rgsaddle, 0.3.11 with eOn"),
    ("minimage", "0.1.2 with rgsaddle, 0.1.4 with eOn"),
    ("rgmin", "0.6.0"),
    ("readcon-core", "0.20.0"),
)

EONCLIB_MISSING = (
    "pyeonclient needs libeonclib loaded before the extension imports. "
    "The harness looks for libeonclib next to the pyeonclient package on "
    "--pythonpath and in --libdir. Put that library in one of those places, "
    "or set LD_LIBRARY_PATH to the directory that contains it."
)

BUILD_NOTES = (
    "eOn was built with native architecture off and fast math off. "
    "Its bundled vector library is not on the timed potential path. "
    "The eOn rgpot engine calls capnp directly, so that client was linked with capnp-rpc.",
)


def eon_package_dirs(pythonpath):
    found = []
    for entry in pythonpath:
        package = os.path.join(entry, "pyeonclient")
        if os.path.isdir(package):
            found.append(package)
        if os.path.basename(os.path.normpath(entry)) == "pyeonclient" and os.path.isdir(entry):
            found.append(entry)
    return found


def _eonclib_name(name):
    return (
        name == "libeonclib.so"
        or name.startswith("libeonclib.so.")
        or name in ("libeonclib.dylib", "eonclib.dll")
    )


def locate_eonclib(pythonpath, libdir):
    roots = []
    if libdir:
        roots.append(libdir)
    for package in eon_package_dirs(pythonpath):
        roots.append(package)
        parent = os.path.dirname(package)
        roots.append(parent)
        roots.append(os.path.join(parent, "lib"))
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if directory:
            roots.append(directory)
    seen = set()
    for root in roots:
        key = os.path.normpath(root)
        if key in seen or not os.path.isdir(root):
            continue
        seen.add(key)
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in sorted(names):
            if _eonclib_name(name):
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    return path
    return None


def preload_eonclib(pythonpath, libdir):
    path = locate_eonclib(pythonpath, libdir)
    if path is None:
        raise SystemExit(EONCLIB_MISSING)
    directory = os.path.dirname(os.path.abspath(path))
    current = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
    if directory not in current:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([directory, *current])
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    return path


def host_cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return sanitize(value)
    except OSError:
        pass
    return "unknown"


def tool_line(command):
    try:
        out = subprocess.check_output(
            [command, "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    line = out.splitlines()[0].strip() if out else ""
    return sanitize(line) if line else None


def sanitize(text):
    out = []
    for token in str(text).split():
        if token.startswith("/") or token.startswith("\\"):
            out.append("<path>")
        else:
            out.append(token)
    return " ".join(out)


def plain(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def dump_json(path, payload):
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(plain(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


class Meter:
    def __init__(self):
        self.calls = 0
        self.kernel_s = 0.0
        self.lock = threading.Lock()

    def add(self, dt):
        with self.lock:
            self.calls += 1
            self.kernel_s += dt

    def snap(self):
        with self.lock:
            return self.calls, self.kernel_s

    def reset(self):
        with self.lock:
            self.calls = 0
            self.kernel_s = 0.0


METERS = {}
MeteredCalculator = None


def meter_for(potential):
    key = id(potential)
    if key not in METERS:
        METERS[key] = Meter()
    return METERS[key]


def bind_calculator():
    global MeteredCalculator
    from rgpot.ase import RgpotCalculator

    class _Metered(RgpotCalculator):
        def __init__(self, potential, meter, **kwargs):
            self._coverage_meter = meter
            super().__init__(potential, **kwargs)

        def __rg_evaluate__(self, positions, numbers, cell, pbc, *, per_atom=False):
            geom = self._geometry(positions, numbers, cell, pbc)
            started = time.perf_counter()
            if per_atom and self._per_atom:
                raw = self._potential.evaluate_per_atom(
                    geom.positions, geom.numbers, geom.box
                )
            else:
                raw = self._potential.evaluate(geom.positions, geom.numbers, geom.box)
            self._coverage_meter.add(time.perf_counter() - started)
            return geom.finish(raw)

    MeteredCalculator = _Metered


class Version(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class BandConfig(ctypes.Structure):
    _fields_ = [
        ("version", Version),
        ("flags", ctypes.c_uint64),
        ("tangent", ctypes.c_int32),
        ("spring", ctypes.c_int32),
        ("projection", ctypes.c_int32),
        ("method", ctypes.c_int32),
        ("spring_k", ctypes.c_double),
        ("spring_ks", ctypes.c_void_p),
        ("ci_trigger_factor", ctypes.c_double),
        ("ci_trigger_force", ctypes.c_double),
        ("cell", ctypes.POINTER(ctypes.c_double)),
        ("force_tol", ctypes.c_double),
        ("max_move", ctypes.c_double),
        ("memory", ctypes.c_int64),
    ]


class MinModeConfig(ctypes.Structure):
    _fields_ = [
        ("version", Version),
        ("flags", ctypes.c_uint64),
        ("kind", ctypes.c_int32),
        ("method", ctypes.c_int32),
        ("dr", ctypes.c_double),
        ("rotation_tol", ctypes.c_double),
        ("max_rotations", ctypes.c_int64),
        ("krylov_dim", ctypes.c_int64),
        ("force_tol", ctypes.c_double),
        ("max_move", ctypes.c_double),
        ("rotation_angle_tol", ctypes.c_double),
        ("difference", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class Report(ctypes.Structure):
    _fields_ = [
        ("version", Version),
        ("flags", ctypes.c_uint64),
        ("status", ctypes.c_int32),
        ("evaluations", ctypes.c_int32),
        ("max_force", ctypes.c_double),
        ("ci_index", ctypes.c_int64),
        ("iteration", ctypes.c_int64),
        ("curvature", ctypes.c_double),
        ("rotations", ctypes.c_int64),
    ]


class SurfaceRequest(ctypes.Structure):
    _fields_ = [
        ("version", Version),
        ("flags", ctypes.c_uint64),
        ("n_images", ctypes.c_int64),
        ("n_atoms", ctypes.c_int64),
        ("positions", ctypes.POINTER(ctypes.c_double)),
        ("energies", ctypes.POINTER(ctypes.c_double)),
        ("gradients", ctypes.POINTER(ctypes.c_double)),
        ("image", ctypes.c_int64),
    ]


class SaddleIndex(ctypes.Structure):
    _fields_ = [
        ("version", Version),
        ("flags", ctypes.c_uint64),
        ("certified", ctypes.c_int32),
        ("index", ctypes.c_int32),
        ("converged", ctypes.c_int32),
        ("n_curvatures", ctypes.c_int32),
        ("evaluations", ctypes.c_int64),
        ("total_evaluations", ctypes.c_int64),
        ("escapes", ctypes.c_int64),
    ]


SURFACE = ctypes.CFUNCTYPE(
    ctypes.c_int32, ctypes.py_object, ctypes.POINTER(SurfaceRequest)
)


def _status_name(lib, code):
    name = lib.rgsaddle_status_name(ctypes.c_int32(code))
    if not name:
        return str(code)
    return name.decode()


class SaddleLibrary:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        lib = self.lib
        lib.rgsaddle_abi_version.restype = ctypes.c_int
        lib.rgsaddle_status_name.argtypes = [ctypes.c_int32]
        lib.rgsaddle_status_name.restype = ctypes.c_char_p
        lib.rgsaddle_band_create.argtypes = [
            ctypes.POINTER(BandConfig),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.rgsaddle_band_create.restype = ctypes.c_void_p
        lib.rgsaddle_band_step.argtypes = [
            ctypes.c_void_p,
            SURFACE,
            ctypes.py_object,
            ctypes.POINTER(Report),
        ]
        lib.rgsaddle_band_step.restype = ctypes.c_int32
        lib.rgsaddle_band_positions.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.rgsaddle_band_positions.restype = ctypes.c_int32
        lib.rgsaddle_band_evaluation.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.rgsaddle_band_evaluation.restype = ctypes.c_int32
        lib.rgsaddle_band_set_fixed_atoms.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
        ]
        lib.rgsaddle_band_set_fixed_atoms.restype = ctypes.c_int32
        lib.rgsaddle_band_set_force_gate.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.rgsaddle_band_set_force_gate.restype = ctypes.c_int32
        lib.rgsaddle_band_saddle_index.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(SaddleIndex),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
        ]
        lib.rgsaddle_band_saddle_index.restype = ctypes.c_int32
        lib.rgsaddle_band_free.argtypes = [ctypes.c_void_p]
        lib.rgsaddle_band_free.restype = None
        lib.rgsaddle_minmode_create.argtypes = [
            ctypes.POINTER(MinModeConfig),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.rgsaddle_minmode_create.restype = ctypes.c_void_p
        lib.rgsaddle_minmode_step.argtypes = [
            ctypes.c_void_p,
            SURFACE,
            ctypes.py_object,
            ctypes.POINTER(Report),
        ]
        lib.rgsaddle_minmode_step.restype = ctypes.c_int32
        lib.rgsaddle_minmode_position.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.rgsaddle_minmode_position.restype = ctypes.c_int32
        lib.rgsaddle_minmode_mode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.rgsaddle_minmode_mode.restype = ctypes.c_int32
        lib.rgsaddle_minmode_set_position.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.rgsaddle_minmode_set_position.restype = ctypes.c_int32
        lib.rgsaddle_minmode_set_mode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.rgsaddle_minmode_set_mode.restype = ctypes.c_int32
        lib.rgsaddle_minmode_set_force_gate.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.rgsaddle_minmode_set_force_gate.restype = ctypes.c_int32
        lib.rgsaddle_minmode_free.argtypes = [ctypes.c_void_p]
        lib.rgsaddle_minmode_free.restype = None
        packed = lib.rgsaddle_abi_version()
        major, minor = (packed >> 16) & 0xFFFF, packed & 0xFFFF
        if major != ABI_MAJOR or minor < ABI_MINOR:
            raise RuntimeError(
                f"rgsaddle abi is {major}.{minor}, this harness wants {ABI_MAJOR}.{ABI_MINOR}"
            )
        self.abi = f"{major}.{minor}"

    def status(self, code):
        return _status_name(self.lib, code)


class Surface:
    def __init__(self, template, calculator, fixed):
        import numpy as np

        self.np = np
        self.template = template
        self.calculator = calculator
        self.fixed = np.asarray(fixed, dtype=bool)
        self.error = ""

    def __call__(self, _self, request):
        try:
            return self.evaluate(request)
        except Exception as exc:
            self.error = sanitize(exc)
            return -6

    def evaluate(self, request):
        np = self.np
        body = request.contents
        n_images = int(body.n_images)
        n_atoms = int(body.n_atoms)
        positions = np.ctypeslib.as_array(
            body.positions, shape=(n_images * n_atoms * 3,)
        ).reshape(n_images, n_atoms, 3)
        energies = np.ctypeslib.as_array(body.energies, shape=(n_images,))
        gradients = np.ctypeslib.as_array(
            body.gradients, shape=(n_images * n_atoms * 3,)
        ).reshape(n_images, n_atoms, 3)
        for image in range(n_images):
            atoms = self.template.copy()
            atoms.positions = np.array(positions[image], dtype=np.float64, copy=True)
            atoms.calc = self.calculator
            forces = atoms.get_forces()
            energies[image] = atoms.get_potential_energy()
            gradient = -np.asarray(forces, dtype=np.float64)
            gradient[self.fixed] = 0.0
            gradients[image] = gradient
        return 0


def band_config(climbing, force_tol, cell):
    config = BandConfig()
    config.version = Version(ABI_MAJOR, ABI_MINOR)
    config.flags = FLAG_NO_MIN_MODE | (FLAG_CI_AFTER if climbing else 0)
    config.tangent = 1
    config.spring = 0
    config.projection = 1
    config.method = 0
    config.spring_k = PROTOCOL["spring_k_ev_per_angstrom2"]
    config.spring_ks = None
    config.ci_trigger_factor = 0.0
    config.ci_trigger_force = 0.5 if climbing else 0.0
    config.cell = cell
    config.force_tol = force_tol
    config.max_move = 0.2
    config.memory = 0
    return config


def minmode_config(kind, force_tol):
    config = MinModeConfig()
    config.version = Version(ABI_MAJOR, ABI_MINOR)
    config.flags = 0
    config.kind = kind
    config.method = 0
    config.dr = 1.0e-3
    config.rotation_tol = 1.0e-4
    config.max_rotations = 20
    config.krylov_dim = 12
    config.force_tol = force_tol
    config.max_move = 0.2
    config.rotation_angle_tol = PROTOCOL["rgsaddle_minmode"]["rotation_angle_tol_radian"]
    config.difference = 0
    config.reserved = 0
    return config


def require_ok(library, code, what):
    if code != 0:
        raise RuntimeError(f"{what} returned {library.status(code)}")


def run_rgsaddle_band(library, system, positions, climbing, force_tol, max_steps, calculator):
    import numpy as np

    n_images, n_atoms, _ = positions.shape
    flat = np.ascontiguousarray(positions, dtype=np.float64)
    cell = None
    cell_buf = None
    if system["periodic"]:
        cell_buf = np.ascontiguousarray(system["cell"], dtype=np.float64)
        cell = cell_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    config = band_config(climbing, force_tol, cell)
    session = library.lib.rgsaddle_band_create(
        ctypes.byref(config),
        ctypes.c_int64(n_images),
        ctypes.c_int64(n_atoms),
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not session:
        raise RuntimeError("band create returned a null session")
    try:
        code = library.lib.rgsaddle_band_set_force_gate(
            session, ctypes.c_int32(GATE_MAX_ATOM)
        )
        require_ok(library, code, "band force gate")
        fixed = np.ascontiguousarray(system["fixed"], dtype=np.uint8)
        if int(fixed.sum()) > 0:
            code = library.lib.rgsaddle_band_set_fixed_atoms(
                session,
                fixed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_int64(n_atoms),
            )
            require_ok(library, code, "band fixed atoms")
        surface = Surface(system["template"], calculator, system["fixed"])
        callback = SURFACE(surface)
        report = Report()
        status = "running"
        started = timed(calculator)
        for _ in range(max_steps):
            code = library.lib.rgsaddle_band_step(
                session, callback, ctypes.py_object(surface), ctypes.byref(report)
            )
            if surface.error:
                status = surface.error
                break
            if code == STALLED:
                status = "stalled"
                break
            if code != 0:
                status = library.status(code)
                break
            if int(report.status) == 1:
                status = "converged"
                break
        else:
            status = "max_iterations"
        wall, calls, kernel = timed_done(calculator, started)
        out = np.empty(n_images * n_atoms * 3, dtype=np.float64)
        code = library.lib.rgsaddle_band_positions(
            session, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        require_ok(library, code, "band positions")
        band = out.reshape(n_images, n_atoms, 3).copy()
        energies = np.empty(n_images, dtype=np.float64)
        code = library.lib.rgsaddle_band_evaluation(
            session,
            energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None,
            None,
        )
        cached = energies.copy() if code == 0 else None
        certificate = None
        if status == "converged" and climbing:
            index = SaddleIndex()
            curvatures = np.zeros(8, dtype=np.float64)
            code = library.lib.rgsaddle_band_saddle_index(
                session,
                ctypes.byref(index),
                curvatures.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(8),
            )
            if code == 0:
                certificate = {
                    "certified": int(index.certified),
                    "index": int(index.index),
                    "converged": int(index.converged),
                }
        return {
            "band": band,
            "cached_energies": cached,
            "ci_index": int(report.ci_index),
            "iteration": int(report.iteration),
            "max_force": float(report.max_force),
            "curvature": float(report.curvature),
            "status": status,
            "certificate": certificate,
            "wall_included": wall,
            "force_calls": calls,
            "kernel_s": kernel,
            "_keep": (config, cell_buf, callback, surface),
        }
    finally:
        library.lib.rgsaddle_band_free(session)


def run_rgsaddle_minmode(library, system, position, mode, kind, calculator):
    import numpy as np

    n_atoms = position.shape[0]
    pos = np.ascontiguousarray(position, dtype=np.float64)
    guess = np.ascontiguousarray(mode, dtype=np.float64)
    config = minmode_config(kind, PROTOCOL["fmax_ev_per_angstrom"])
    session = library.lib.rgsaddle_minmode_create(
        ctypes.byref(config),
        ctypes.c_int64(n_atoms),
        pos.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        guess.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not session:
        raise RuntimeError("min-mode create returned a null session")
    fixed = np.asarray(system["fixed"], dtype=bool)
    anchor = np.array(system["reactant"], dtype=np.float64)
    try:
        code = library.lib.rgsaddle_minmode_set_force_gate(
            session, ctypes.c_int32(GATE_MAX_ATOM)
        )
        require_ok(library, code, "min-mode force gate")
        surface = Surface(system["template"], calculator, system["fixed"])
        callback = SURFACE(surface)
        report = Report()
        status = "running"
        holder = pos.copy()
        started = timed(calculator)
        for _ in range(PROTOCOL["max_iterations"]):
            code = library.lib.rgsaddle_minmode_step(
                session, callback, ctypes.py_object(surface), ctypes.byref(report)
            )
            if surface.error:
                status = surface.error
                break
            if code != 0:
                status = library.status(code)
                break
            if fixed.any():
                code = library.lib.rgsaddle_minmode_position(
                    session, holder.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                )
                require_ok(library, code, "min-mode position")
                current = holder.reshape(n_atoms, 3)
                if not np.allclose(current[fixed], anchor[fixed]):
                    current[fixed] = anchor[fixed]
                    mode_buf = np.empty(n_atoms * 3, dtype=np.float64)
                    library.lib.rgsaddle_minmode_mode(
                        session,
                        mode_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    )
                    mode_atoms = mode_buf.reshape(n_atoms, 3)
                    mode_atoms[fixed] = 0.0
                    norm = np.linalg.norm(mode_atoms)
                    if norm > 0.0:
                        mode_atoms /= norm
                        library.lib.rgsaddle_minmode_set_mode(
                            session,
                            mode_atoms.reshape(-1).ctypes.data_as(
                                ctypes.POINTER(ctypes.c_double)
                            ),
                        )
                    library.lib.rgsaddle_minmode_set_position(
                        session,
                        current.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        None,
                    )
            if int(report.status) == 1:
                status = "converged"
                break
        else:
            status = "max_iterations"
        wall, calls, kernel = timed_done(calculator, started)
        out = np.empty(n_atoms * 3, dtype=np.float64)
        code = library.lib.rgsaddle_minmode_position(
            session, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        require_ok(library, code, "min-mode final position")
        final = out.reshape(n_atoms, 3).copy()
        if fixed.any():
            final[fixed] = anchor[fixed]
        return {
            "position": final,
            "iteration": int(report.iteration),
            "max_force": float(report.max_force),
            "curvature": float(report.curvature),
            "status": status,
            "wall_included": wall,
            "force_calls": calls,
            "kernel_s": kernel,
            "_keep": (config, callback, surface),
        }
    finally:
        library.lib.rgsaddle_minmode_free(session)


def new_calculator(potential):
    return MeteredCalculator(potential, meter_for(potential))


def timed(calculator):
    calculator._coverage_meter.reset()
    return time.perf_counter()


def timed_done(calculator, started):
    calls, kernel = calculator._coverage_meter.snap()
    return time.perf_counter() - started, calls, kernel


def probe(template, potential, positions, fixed):
    import numpy as np

    atoms = template.copy()
    atoms.positions = np.asarray(positions, dtype=np.float64)
    atoms.calc = new_calculator(potential)
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    free = ~np.asarray(fixed, dtype=bool)
    norms = np.linalg.norm(forces[free], axis=1) if free.any() else np.zeros(0)
    fmax = float(norms.max()) if norms.size else 0.0
    return float(atoms.get_potential_energy()), fmax


def linear_band(reactant, product, n_images):
    import numpy as np

    band = np.empty((n_images, reactant.shape[0], 3), dtype=np.float64)
    for image in range(n_images):
        weight = image / (n_images - 1)
        band[image] = (1.0 - weight) * reactant + weight * product
    return band


def select_image(band, energies, ci_index, climbing):
    import numpy as np

    if climbing and ci_index is not None and 0 < ci_index < len(band) - 1:
        return int(ci_index)
    interior = np.asarray(energies[1:-1], dtype=float)
    return 1 + int(np.argmax(interior))


class MiCell(ctypes.Structure):
    _fields_ = [
        ("ax", ctypes.c_double),
        ("ay", ctypes.c_double),
        ("az", ctypes.c_double),
        ("bx", ctypes.c_double),
        ("by", ctypes.c_double),
        ("bz", ctypes.c_double),
        ("cx", ctypes.c_double),
        ("cy", ctypes.c_double),
        ("cz", ctypes.c_double),
        ("ox", ctypes.c_double),
        ("oy", ctypes.c_double),
        ("oz", ctypes.c_double),
    ]


_MINIMAGE = None


def bind_minimage(library):
    global _MINIMAGE
    lib = library.lib
    lib.minimage_cell_from_ase.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(MiCell),
    ]
    lib.minimage_cell_from_ase.restype = ctypes.c_int
    lib.minimage_displacement_euclidean.argtypes = [
        ctypes.POINTER(MiCell),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.minimage_displacement_euclidean.restype = ctypes.c_int
    _MINIMAGE = lib


def mic_vectors(origin, dest, cell):
    import numpy as np

    if _MINIMAGE is None:
        raise RuntimeError("minimum image is not bound")
    origin = np.ascontiguousarray(origin, dtype=np.float64)
    dest = np.ascontiguousarray(dest, dtype=np.float64)
    rows = np.ascontiguousarray(np.asarray(cell, dtype=np.float64).reshape(-1))
    box = MiCell()
    code = _MINIMAGE.minimage_cell_from_ase(
        rows.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        None,
        ctypes.byref(box),
    )
    if code != 0:
        raise RuntimeError(f"minimage_cell_from_ase returned {code}")
    out = np.empty_like(origin)
    dr = np.empty(3, dtype=np.float64)
    for index in range(origin.shape[0]):
        code = _MINIMAGE.minimage_displacement_euclidean(
            ctypes.byref(box),
            origin[index].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dest[index].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if code != 0:
            raise RuntimeError(f"minimage_displacement_euclidean returned {code}")
        out[index] = dr
    return out


def _kabsch_rmsd(reference, moving):
    import numpy as np

    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if reference.shape != moving.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise RuntimeError("rmsd coordinates do not match")
    if reference.shape[0] < 2:
        delta = moving - reference
        if reference.shape[0] == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    left = reference - reference.mean(axis=0)
    right = moving - moving.mean(axis=0)
    covariance = left.T @ right
    left_basis, _singular, right_basis = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(right_basis.T @ left_basis.T))
    if correction[-1, -1] == 0.0:
        correction[-1, -1] = 1.0
    rotation = right_basis.T @ correction @ left_basis.T
    aligned = right @ rotation
    delta = left - aligned
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _species_rmsd(reference, trial, numbers):
    from itertools import permutations, product

    labels = [int(number) for number in numbers]
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    blocks = [indexes for indexes in groups.values() if len(indexes) > 1]
    total = 1
    for indexes in blocks:
        total *= math.factorial(len(indexes))
        if total > PERMUTATION_CAP:
            return _kabsch_rmsd(reference, trial)
    if not blocks:
        return _kabsch_rmsd(reference, trial)
    best = None
    choices = [list(permutations(indexes)) for indexes in blocks]
    for combo in product(*choices):
        order = list(range(len(labels)))
        for indexes, perm in zip(blocks, combo):
            for src, dst in zip(indexes, perm):
                order[src] = dst
        value = _kabsch_rmsd(reference, trial[order])
        if best is None or value < best:
            best = value
    return best


def rmsd(system, reference, trial):
    import numpy as np

    free = ~np.asarray(system["fixed"], dtype=bool)
    if not free.any():
        raise RuntimeError("every atom is fixed")
    left = np.asarray(reference, dtype=np.float64)[free]
    right = np.asarray(trial, dtype=np.float64)[free]
    if system["periodic"]:
        delta = mic_vectors(left, right, np.asarray(system["cell"], dtype=np.float64))
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    numbers = system.get("numbers")
    if numbers is None:
        return _kabsch_rmsd(left, right)
    return _species_rmsd(left, right, np.asarray(numbers)[free])


def displacement_seed(name, seed):
    # The round index alone is one stream. pt_hop and pt_exchange share a
    # reactant, and a min-mode search never reads the product, so that
    # stream was one trajectory. The name tag gives each system its own block.
    tag = zlib.crc32(str(name).encode("utf-8")) & 0x7FFFFFFF
    return PROTOCOL["displacement_seed_base"] + int(seed) + tag * (PROTOCOL["n_seeds"] + 1)


def displace(system, seed):
    import numpy as np

    rng = np.random.Generator(np.random.PCG64(displacement_seed(system["name"], seed)))
    draw = rng.normal(
        0.0,
        PROTOCOL["displacement_sigma_angstrom"],
        size=np.shape(system["reactant"]),
    )
    fixed = np.asarray(system["fixed"], dtype=bool)
    draw[fixed] = 0.0
    norm = np.linalg.norm(draw)
    if norm == 0.0:
        raise RuntimeError("displacement vanished")
    start = np.asarray(system["reactant"], dtype=np.float64) + draw
    return start, draw / norm


def eon_parameters(pyec, kind):
    params = pyec.Parameters()
    params.opt_convergence_metric = "max_atom"
    params.opt_converged_force = PROTOCOL["fmax_ev_per_angstrom"]
    params.opt_max_iterations = PROTOCOL["max_iterations"]
    params.saddle_converged_force = PROTOCOL["fmax_ev_per_angstrom"]
    params.saddle_max_iterations = PROTOCOL["max_iterations"]
    params.neb_force_tolerance = PROTOCOL["fmax_ev_per_angstrom"]
    params.neb_max_iterations = PROTOCOL["max_iterations"]
    params.neb_init_method = pyec.NEBInit.LINEAR
    params.neb_minimize_endpoints = False
    blob = json.loads(params.to_json())
    spring = blob["Nudged Elastic Band"]["spring"]["constant"]
    if float(spring) != PROTOCOL["spring_k_ev_per_angstrom2"]:
        raise RuntimeError(f"eOn spring constant is {spring}")
    if kind == "neb":
        params.neb_climbing_image = False
        params.neb_images = 0
    elif kind == "ci":
        params.neb_climbing_image = True
    elif kind == "dimer":
        params.saddle_minmode_method = "dimer"
    elif kind == "lanczos":
        params.saddle_minmode_method = "lanczos"
    return params


def eon_matter(pyec, system, positions, calculator, params, eon_pot=None):
    import numpy as np

    atoms = system["template"].copy()
    atoms.positions = np.asarray(positions, dtype=np.float64)
    atoms.calc = calculator
    if eon_pot is None:
        matter = pyec.ase_to_matter(atoms, parameters=params)
    else:
        matter = pyec.ase_to_matter(atoms, potential=eon_pot, parameters=params)
    matter.fixed = np.ascontiguousarray(system["fixed"], dtype=np.int32)
    return matter


def run_eon_neb(pyec, system, n_images, climbing, potential):
    import numpy as np

    params = eon_parameters(pyec, "ci" if climbing else "neb")
    params.neb_images = n_images - 2
    calculator = new_calculator(potential)
    eon_pot = pyec.ase_potential(calculator).potential
    initial = eon_matter(pyec, system, system["reactant"], calculator, params, eon_pot)
    final = eon_matter(pyec, system, system["product"], calculator, params, eon_pot)
    neb = pyec.NEB(initial, final, params, eon_pot)
    started = timed(calculator)
    neb.compute()
    wall, calls, kernel = timed_done(calculator, started)
    band = neb.band
    n_path = int(band.n_path)
    if n_path != n_images:
        raise RuntimeError(f"eOn path has {n_path} images, expected {n_images}")
    if float(band.ksp) != PROTOCOL["spring_k_ev_per_angstrom2"]:
        raise RuntimeError(f"eOn spring in use is {float(band.ksp)}")
    positions = []
    energies = []
    for index in range(n_path):
        image = band.image(index)
        positions.append(np.array(image.positions, dtype=np.float64, copy=True))
        energies.append(float(band.image_energy(index)))
    ci_index = int(band.climbing_image)
    chosen = select_image(positions, energies, ci_index, climbing)
    status = getattr(band.status, "name", str(band.status))
    return {
        "position": positions[chosen],
        "cached_energy": energies[chosen],
        "status": status,
        "converged_flag": band.status == pyec.NEBStatus.GOOD,
        "iteration": None,
        "curvature": None,
        "max_force": float(band.convergence_force()),
        "wall_included": wall,
        "force_calls": calls,
        "kernel_s": kernel,
    }


def run_eon_minmode(pyec, system, position, mode, lanczos, potential, reactant_energy):
    import numpy as np

    params = eon_parameters(pyec, "lanczos" if lanczos else "dimer")
    calculator = new_calculator(potential)
    eon_pot = pyec.ase_potential(calculator).potential
    matter = eon_matter(pyec, system, position, calculator, params, eon_pot)
    mode_array = np.ascontiguousarray(mode, dtype=np.float64)
    started = timed(calculator)
    result, status = pyec.min_mode_saddle_search(
        matter, mode_array, float(reactant_energy), params, eon_pot
    )
    wall, calls, kernel = timed_done(calculator, started)
    final = np.array(result.positions, dtype=np.float64, copy=True)
    return {
        "position": final,
        "status": getattr(status, "name", str(status)),
        "converged_flag": status == pyec.SaddleStatus.GOOD,
        "iteration": None,
        "curvature": None,
        "max_force": None,
        "wall_included": wall,
        "force_calls": calls,
        "kernel_s": kernel,
    }


def ase_images(system, band, potential):
    from ase.constraints import FixAtoms
    import numpy as np

    fixed = np.flatnonzero(np.asarray(system["fixed"], dtype=bool))
    images = []
    for positions in band:
        atoms = system["template"].copy()
        atoms.positions = positions
        atoms.calc = new_calculator(potential)
        if fixed.size:
            atoms.set_constraint(FixAtoms(indices=fixed.tolist()))
        images.append(atoms)
    return images


def run_ase_neb(system, n_images, climbing, potential):
    from ase.mep import NEB
    from ase.optimize import BFGS

    band = linear_band(system["reactant"], system["product"], n_images)
    images = ase_images(system, band, potential)
    neb = NEB(
        images,
        k=PROTOCOL["spring_k_ev_per_angstrom2"],
        climb=climbing,
        method="improvedtangent",
    )
    optimizer = BFGS(neb, logfile=None)
    started = timed(images[0].calc)
    optimizer.run(fmax=PROTOCOL["fmax_ev_per_angstrom"], steps=PROTOCOL["max_iterations"])
    wall, calls, kernel = timed_done(images[0].calc, started)
    energies = [float(image.get_potential_energy()) for image in images]
    positions = [image.positions.copy() for image in images]
    chosen = select_image(positions, energies, None, climbing)
    return {
        "position": positions[chosen],
        "cached_energy": energies[chosen],
        "status": "converged" if optimizer.converged() else "max_iterations",
        "converged_flag": bool(optimizer.converged()),
        "iteration": int(optimizer.nsteps),
        "curvature": None,
        "max_force": None,
        "wall_included": wall,
        "force_calls": calls,
        "kernel_s": kernel,
    }


def run_ase_dimer(system, position, mode, potential):
    import numpy as np
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
    from ase.constraints import FixAtoms

    atoms = system["template"].copy()
    atoms.positions = np.asarray(position, dtype=np.float64)
    atoms.calc = new_calculator(potential)
    fixed = np.asarray(system["fixed"], dtype=bool)
    if fixed.any():
        atoms.set_constraint(FixAtoms(indices=np.flatnonzero(fixed).tolist()))
    control = DimerControl(
        logfile=None,
        eigenmode_logfile=None,
        mask=(~fixed).tolist(),
    )
    dimer = MinModeAtoms(
        atoms,
        control=control,
        eigenmodes=[np.asarray(mode, dtype=np.float64)],
        random_seed=PROTOCOL["displacement_seed_base"],
    )
    optimizer = MinModeTranslate(dimer, logfile=None)
    anchor = np.asarray(system["reactant"], dtype=np.float64)

    def restore():
        current = dimer.get_positions()
        current[fixed] = anchor[fixed]
        dimer.set_positions(current)
        guess = np.array(dimer.get_eigenmode(), dtype=np.float64).reshape(-1, 3)
        guess[fixed] = 0.0
        norm = np.linalg.norm(guess)
        if norm > 0.0:
            dimer.eigenmodes[0] = guess / norm

    if fixed.any():
        optimizer.attach(restore)
    started = timed(atoms.calc)
    optimizer.run(fmax=PROTOCOL["fmax_ev_per_angstrom"], steps=PROTOCOL["max_iterations"])
    wall, calls, kernel = timed_done(atoms.calc, started)
    final = dimer.get_positions().copy()
    if fixed.any():
        final[fixed] = anchor[fixed]
    curvature = None
    try:
        curvature = float(dimer.get_curvature())
    except Exception:
        curvature = None
    return {
        "position": final,
        "status": "converged" if optimizer.converged() else "max_iterations",
        "converged_flag": bool(optimizer.converged()),
        "iteration": int(optimizer.nsteps),
        "curvature": curvature,
        "max_force": None,
        "wall_included": wall,
        "force_calls": calls,
        "kernel_s": kernel,
    }


def finish_record(system, method, round_index, order, images, seed, result, potential):
    wall = float(result["wall_included"])
    search_calls = int(result["force_calls"])
    search_kernel = float(result["kernel_s"])
    position = result["position"]
    energy, fmax = probe(system["template"], potential, position, system["fixed"])
    reference = system["reference"]
    distance = None
    error = result.get("error")
    try:
        distance = rmsd(system, reference["positions"], position)
    except Exception as exc:
        error = sanitize(exc)
    barrier = energy - float(system["reactant_energy"])
    usable = bool(reference.get("usable"))
    converged = bool(result["converged_flag"])
    energy_ok = abs(energy - float(reference["energy"])) <= PROTOCOL["energy_tol_ev"]
    rmsd_ok = distance is not None and distance <= PROTOCOL["rmsd_tol_angstrom"]
    success = bool(usable and converged and energy_ok and rmsd_ok)
    return {
        "system": system["name"],
        "method": method,
        "round": round_index,
        "order": order,
        "images": images,
        "seed": seed,
        "converged": converged,
        "success": success,
        "wall_s": wall,
        "kernel_s": search_kernel,
        "outside_s": wall - search_kernel,
        "force_calls": search_calls,
        "energy": energy,
        "barrier": barrier,
        "energy_error": energy - float(reference["energy"]),
        "barrier_error": barrier - float(reference["barrier"]),
        "rmsd": distance,
        "fmax_free": fmax,
        "iteration": result.get("iteration"),
        "curvature": result.get("curvature"),
        "max_force": result.get("max_force"),
        "status": result.get("status"),
        "error": error,
        "positions": position,
    }


def failed_record(system, method, round_index, order, images, seed, message, calls, kernel):
    return {
        "system": system["name"],
        "method": method,
        "round": round_index,
        "order": order,
        "images": images,
        "seed": seed,
        "converged": False,
        "success": False,
        "wall_s": None,
        "kernel_s": kernel,
        "outside_s": None,
        "force_calls": calls,
        "energy": None,
        "barrier": None,
        "energy_error": None,
        "barrier_error": None,
        "rmsd": None,
        "fmax_free": None,
        "iteration": None,
        "curvature": None,
        "max_force": None,
        "status": "error",
        "error": sanitize(message),
        "positions": None,
    }


def execute_neb(library, pyec, system, method, n_images, potential):
    climbing = method.endswith("-ci")
    if method.startswith("rgsaddle"):
        band = linear_band(system["reactant"], system["product"], n_images)
        calculator = new_calculator(potential)
        outcome = run_rgsaddle_band(
            library,
            system,
            band,
            climbing,
            PROTOCOL["fmax_ev_per_angstrom"],
            PROTOCOL["max_iterations"],
            calculator,
        )
        energies = outcome["cached_energies"]
        if energies is None:
            energies = [
                probe(system["template"], potential, row, system["fixed"])[0]
                for row in outcome["band"]
            ]
        chosen = select_image(outcome["band"], energies, outcome["ci_index"], climbing)
        return {
            "position": outcome["band"][chosen],
            "status": outcome["status"],
            "converged_flag": outcome["status"] == "converged",
            "iteration": outcome["iteration"],
            "curvature": outcome["curvature"],
            "max_force": outcome["max_force"],
            "wall_included": outcome["wall_included"],
            "force_calls": outcome["force_calls"],
            "kernel_s": outcome["kernel_s"],
        }
    if method.startswith("eon"):
        return run_eon_neb(pyec, system, n_images, climbing, potential)
    return run_ase_neb(system, n_images, climbing, potential)


def execute_mm(library, pyec, system, method, seed, potential):
    position, mode = displace(system, seed)
    if method == "rgsaddle-dimer":
        kind = 0
    elif method == "rgsaddle-lanczos":
        kind = 1
    else:
        kind = None
    if kind is not None:
        calculator = new_calculator(potential)
        outcome = run_rgsaddle_minmode(library, system, position, mode, kind, calculator)
        outcome["converged_flag"] = outcome["status"] == "converged"
        return outcome
    if method == "eon-dimer" or method == "eon-lanczos":
        return run_eon_minmode(
            pyec,
            system,
            position,
            mode,
            method.endswith("lanczos"),
            potential,
            system["reactant_energy"],
        )
    return run_ase_dimer(system, position, mode, potential)


def median_iqr(values):
    import numpy as np

    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None, None, 0
    array = np.asarray(finite, dtype=float)
    q1, med, q3 = np.percentile(array, [25, 50, 75])
    return float(med), float(q3 - q1), int(array.size)


def mannwhitney(left, right):
    import numpy as np
    from scipy.stats import mannwhitneyu

    a = [float(value) for value in left if value is not None and math.isfinite(float(value))]
    b = [float(value) for value in right if value is not None and math.isfinite(float(value))]
    if len(a) == 0 or len(b) == 0:
        return None
    if np.all(np.asarray(a) == a[0]) and np.all(np.asarray(b) == b[0]) and a[0] == b[0]:
        return 1.0
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue)
    except ValueError:
        return None


def fisher(success_a, n_a, success_b, n_b):
    from scipy.stats import fisher_exact

    table = [[success_a, n_a - success_a], [success_b, n_b - success_b]]
    return float(fisher_exact(table, alternative="two-sided").pvalue)


def summarize(runs):
    cells = {}
    for run in runs:
        key = (run["system"], run["method"], run["images"])
        cells.setdefault(key, []).append(run)
    summary = []
    for key, rows in cells.items():
        system, method, images = key
        even = [row for row in rows if row["round"] % 2 == 0]
        odd = [row for row in rows if row["round"] % 2 == 1]
        wall_med, wall_iqr, wall_n = median_iqr([row["wall_s"] for row in rows])
        out_med, out_iqr, _ = median_iqr([row["outside_s"] for row in rows])
        call_med, call_iqr, _ = median_iqr([row["force_calls"] for row in rows])
        successes = [row for row in rows if row["success"]]
        bar_med, bar_iqr, bar_n = median_iqr([row["barrier"] for row in successes])
        rms_med, rms_iqr, _ = median_iqr([row["rmsd"] for row in rows])
        err_med, err_iqr, _ = median_iqr(
            [abs(row["energy_error"]) for row in rows if row["energy_error"] is not None]
        )
        summary.append(
            {
                "system": system,
                "method": method,
                "images": images,
                "n": len(rows),
                "n_success": len(successes),
                "success_rate": len(successes) / len(rows) if rows else None,
                "wall_median_s": wall_med,
                "wall_iqr_s": wall_iqr,
                "outside_median_s": out_med,
                "outside_iqr_s": out_iqr,
                "force_calls_median": call_med,
                "force_calls_iqr": call_iqr,
                "barrier_median_ev": bar_med,
                "barrier_iqr_ev": bar_iqr,
                "barrier_n": bar_n,
                "rmsd_median_angstrom": rms_med,
                "rmsd_iqr_angstrom": rms_iqr,
                "abs_energy_error_median_ev": err_med,
                "abs_energy_error_iqr_ev": err_iqr,
                "aa_wall_p": mannwhitney(
                    [row["wall_s"] for row in even], [row["wall_s"] for row in odd]
                ),
                "aa_calls_p": mannwhitney(
                    [row["force_calls"] for row in even],
                    [row["force_calls"] for row in odd],
                ),
                "aa_success_p": fisher(
                    sum(row["success"] for row in even),
                    len(even),
                    sum(row["success"] for row in odd),
                    len(odd),
                )
                if even and odd
                else None,
                "wall_n": wall_n,
            }
        )
    comparisons = []
    pairs = (
        ("rgsaddle-neb", "eon-neb"),
        ("rgsaddle-neb", "ase-neb"),
        ("rgsaddle-ci", "eon-ci"),
        ("rgsaddle-ci", "ase-ci"),
        ("rgsaddle-dimer", "eon-dimer"),
        ("rgsaddle-dimer", "ase-dimer"),
        ("rgsaddle-lanczos", "eon-lanczos"),
    )
    by_key = {key: rows for key, rows in cells.items()}
    systems = sorted({row["system"] for row in runs})
    image_values = sorted({row["images"] for row in runs}, key=lambda item: (item is None, item or 0))
    for system in systems:
        for images in image_values:
            for left_name, right_name in pairs:
                left = by_key.get((system, left_name, images))
                right = by_key.get((system, right_name, images))
                if not left or not right:
                    continue
                comparisons.append(
                    {
                        "system": system,
                        "images": images,
                        "a": left_name,
                        "b": right_name,
                        "wall_p": mannwhitney(
                            [row["wall_s"] for row in left], [row["wall_s"] for row in right]
                        ),
                        "outside_p": mannwhitney(
                            [row["outside_s"] for row in left],
                            [row["outside_s"] for row in right],
                        ),
                        "calls_p": mannwhitney(
                            [row["force_calls"] for row in left],
                            [row["force_calls"] for row in right],
                        ),
                        "barrier_p": mannwhitney(
                            [row["barrier"] for row in left if row["success"]],
                            [row["barrier"] for row in right if row["success"]],
                        ),
                        "success_p": fisher(
                            sum(row["success"] for row in left),
                            len(left),
                            sum(row["success"] for row in right),
                            len(right),
                        ),
                    }
                )
    return summary, comparisons


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_table(path, payload):
    lines = [
        "# Saddle-search coverage",
        "",
        "Stop when the maximum per-atom force norm is under 0.01 eV/Angstrom.",
        "Spring 5.0 eV/Angstrom^2, linear path, 7 and 11 images, 1000 iterations.",
        "Success: code converged, reference index is 1, energy within 0.01 eV,",
        "free-atom RMSD within 0.05 Angstrom. A free cluster is centered and",
        "Kabsch-aligned, and identical species are matched. Medians and IQRs use n=12.",
        "A/A is a Mann-Whitney test of even rounds against odd rounds.",
        "Method p-values are Mann-Whitney, and success p-values are Fisher exact.",
        "ASE ships no Lanczos min-mode search.",
        "",
    ]
    systems = []
    for row in payload["summary"]:
        if row["system"] not in systems:
            systems.append(row["system"])
    for system in systems:
        lines.append(f"## {system}")
        lines.append("")
        reference = payload["systems"][system]["reference"]
        lines.append(
            "Reference usable: {usable}; index {index}; barrier {barrier} eV.".format(
                usable=reference.get("usable"),
                index=reference.get("index"),
                barrier=fmt(reference.get("barrier")),
            )
        )
        lines.append("")
        lines.append(
            "| method | images | success | wall s (IQR) | outside s (IQR) | calls (IQR) | barrier eV (IQR) | rmsd A | A/A wall p |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in payload["summary"]:
            if row["system"] != system:
                continue
            images = "" if row["images"] is None else str(row["images"])
            lines.append(
                "| {method} | {images} | {ok}/{n} | {wall} ({wi}) | {out} ({oi}) | {calls} ({ci}) | {bar} ({bi}) | {rmsd} | {aa} |".format(
                    method=row["method"],
                    images=images,
                    ok=row["n_success"],
                    n=row["n"],
                    wall=fmt(row["wall_median_s"]),
                    wi=fmt(row["wall_iqr_s"]),
                    out=fmt(row["outside_median_s"]),
                    oi=fmt(row["outside_iqr_s"]),
                    calls=fmt(row["force_calls_median"]),
                    ci=fmt(row["force_calls_iqr"]),
                    bar=fmt(row["barrier_median_ev"]),
                    bi=fmt(row["barrier_iqr_ev"]),
                    rmsd=fmt(row["rmsd_median_angstrom"]),
                    aa=fmt(row["aa_wall_p"]),
                )
            )
        lines.append("")
        lines.append("| comparison | images | wall p | outside p | calls p | barrier p | success p |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in payload["comparisons"]:
            if row["system"] != system:
                continue
            images = "" if row["images"] is None else str(row["images"])
            lines.append(
                "| {a} vs {b} | {images} | {wall} | {out} | {calls} | {bar} | {suc} |".format(
                    a=row["a"],
                    b=row["b"],
                    images=images,
                    wall=fmt(row["wall_p"]),
                    out=fmt(row["outside_p"]),
                    calls=fmt(row["calls_p"]),
                    bar=fmt(row["barrier_p"]),
                    suc=fmt(row["success_p"]),
                )
            )
        lines.append("")
    lines.append("## Failures")
    lines.append("")
    failures = [row for row in payload["runs"] if row["status"] not in ("converged",) and not row["success"]]
    if not failures:
        lines.append("No failed runs.")
    else:
        lines.append("| system | method | images | round | status |")
        lines.append("| --- | --- | --- | --- | --- |")
        shown = failures[:40]
        for row in shown:
            images = "" if row["images"] is None else str(row["images"])
            lines.append(
                f"| {row['system']} | {row['method']} | {images} | {row['round']} | {row['status']} |"
            )
        if len(failures) > len(shown):
            lines.append("")
            lines.append(f"{len(failures) - len(shown)} further failures are in the JSON.")
    lines.append("")
    lines.extend(build_section(payload))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def build_section(payload):
    versions = payload.get("versions") or {}
    shas = payload.get("shas") or {}
    lines = ["## Build", ""]
    lines.append("| component | revision or version |")
    lines.append("| --- | --- |")
    abi = versions.get("rgsaddle_abi")
    rgsaddle_parts = [shas.get("rgsaddle") or ""]
    if abi:
        rgsaddle_parts.append(f"ABI {abi}")
    rgsaddle_parts.append(RGSADDLE_BUILD)
    rgsaddle = ", ".join(part for part in rgsaddle_parts if part)
    if rgsaddle:
        lines.append(f"| rgsaddle | {rgsaddle} |")
    rgpot_parts = [shas.get("rgpot") or "", versions.get("rgpot") or "", RGPOT_BUILD]
    rgpot = ", ".join(part for part in rgpot_parts if part)
    if rgpot:
        lines.append(f"| rgpot | {rgpot} |")
    eon_parts = [shas.get("eon") or ""]
    if versions.get("pyeonclient"):
        eon_parts.append(f"pyeonclient {versions['pyeonclient']}")
    project = versions.get("eon_project") or EON_PROJECT_VERSION
    if project:
        eon_parts.append(f"project version {project}")
    eon = ", ".join(part for part in eon_parts if part)
    if eon:
        lines.append(f"| eOn | {eon} |")
    for key, label in (
        ("ase", "ASE"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("rustc", "rustc"),
        ("gcc", "gcc"),
        ("g++", "g++"),
    ):
        if versions.get(key):
            lines.append(f"| {label} | {versions[key]} |")
    for name, value in BUILD_PINS:
        lines.append(f"| {name} | {value} |")
    cpu = payload.get("cpu_model")
    if cpu:
        lines.append(f"| cpu | {cpu} |")
    lines.append("")
    for note in BUILD_NOTES:
        lines.append(note)
    threads = payload.get("threads")
    rounds = payload.get("rounds")
    if threads is not None and rounds is not None:
        if int(threads) == 1:
            lines.append(f"One core, one thread, {rounds} rounds.")
        else:
            lines.append(f"One core, {threads} threads, {rounds} rounds.")
    lines.append("")
    return lines


def lattice_constant(potential, atomic_number):
    import numpy as np

    def energy(length):
        cell = np.eye(3) * length
        basis = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
            dtype=np.float64,
        )
        positions = np.ascontiguousarray(basis * length)
        numbers = np.full(4, atomic_number, dtype=np.int32)
        return float(potential.evaluate(positions, numbers, cell)["energy"])

    low, high = 3.5, 4.8
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - golden * (high - low)
    right = low + golden * (high - low)
    left_energy, right_energy = energy(left), energy(right)
    for _ in range(48):
        if left_energy < right_energy:
            high, right, right_energy = right, left, left_energy
            left = high - golden * (high - low)
            left_energy = energy(left)
        else:
            low, left, left_energy = left, right, right_energy
            right = low + golden * (high - low)
            right_energy = energy(right)
    return 0.5 * (low + high)


def relax(template, potential, positions, fixed, fmax, steps):
    import numpy as np
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS

    atoms = template.copy()
    atoms.positions = np.asarray(positions, dtype=np.float64)
    atoms.calc = new_calculator(potential)
    mask = np.flatnonzero(np.asarray(fixed, dtype=bool))
    if mask.size:
        atoms.set_constraint(FixAtoms(indices=mask.tolist()))
    optimizer = BFGS(atoms, logfile=None)
    optimizer.run(fmax=fmax, steps=steps)
    return (
        atoms.get_positions().copy(),
        float(atoms.get_potential_energy()),
        bool(optimizer.converged()),
    )


def make_template(numbers, positions, cell, pbc):
    from ase import Atoms
    import numpy as np

    return Atoms(
        numbers=np.asarray(numbers, dtype=int),
        positions=np.asarray(positions, dtype=float),
        cell=np.asarray(cell, dtype=float),
        pbc=pbc,
    )


def pack_system(name, potential_name, periodic, numbers, cell, pbc, fixed, reactant, product, reactant_energy, product_energy, extra):
    import numpy as np

    reactant = np.asarray(reactant, dtype=np.float64)
    product = np.asarray(product, dtype=np.float64)
    fixed = np.asarray(fixed, dtype=np.uint8)
    product = product.copy()
    product[fixed.astype(bool)] = reactant[fixed.astype(bool)]
    system = {
        "name": name,
        "potential": potential_name,
        "periodic": bool(periodic),
        "numbers": [int(number) for number in numbers],
        "cell": np.asarray(cell, dtype=np.float64).tolist(),
        "pbc": [bool(flag) for flag in pbc],
        "fixed": [int(flag) for flag in fixed],
        "reactant": reactant.tolist(),
        "product": product.tolist(),
        "reactant_energy": float(reactant_energy),
        "product_energy": float(product_energy),
    }
    system.update(extra)
    return system


def distinct(system):
    probe = {
        "fixed": system["fixed"],
        "periodic": system["periodic"],
        "cell": system["cell"],
    }
    if "numbers" in system:
        probe["numbers"] = system["numbers"]
    distance = rmsd(probe, system["reactant"], system["product"])
    if distance < 0.2:
        raise RuntimeError(f"{system['name']} endpoints are {distance} Angstrom apart")
    return distance


def build_pt(morse, lattice):
    import numpy as np
    from ase.build import add_adsorbate, fcc111

    slab = fcc111(
        "Pt", size=(3, 3, 3), a=lattice, vacuum=PROTOCOL["pt_vacuum_angstrom"]
    )
    slab.set_pbc(True)
    normal = np.cross(slab.cell[0], slab.cell[1])
    height = abs(float(np.dot(slab.cell[2], normal) / np.linalg.norm(normal)))
    cutoff = float(morse.config()["cutoff"])
    if height <= 2.0 * cutoff:
        raise RuntimeError(f"Pt cell height {height} does not clear two cutoffs")
    z = slab.positions[:, 2]
    fixed = z < (z.min() + 0.5)
    height = 2.0
    hop_start = slab.copy()
    hop_end = slab.copy()
    add_adsorbate(hop_start, "Pt", height, "fcc")
    add_adsorbate(hop_end, "Pt", height, "hcp")
    numbers = hop_start.numbers.copy()
    cell = hop_start.cell.array.copy()
    pbc = hop_start.pbc.copy()
    fixed = np.concatenate([fixed, [False]])
    template = make_template(numbers, hop_start.positions, cell, pbc)
    reactant, reactant_energy, reactant_ok = relax(
        template,
        morse,
        hop_start.positions,
        fixed,
        PROTOCOL["endpoint_fmax_ev_per_angstrom"],
        PROTOCOL["endpoint_max_iterations"],
    )
    product, product_energy, product_ok = relax(
        template,
        morse,
        hop_end.positions,
        fixed,
        PROTOCOL["endpoint_fmax_ev_per_angstrom"],
        PROTOCOL["endpoint_max_iterations"],
    )
    hop = pack_system(
        "pt_hop",
        "morse",
        True,
        numbers,
        cell,
        pbc,
        fixed,
        reactant,
        product,
        reactant_energy,
        product_energy,
        {
            "lattice_angstrom": lattice,
            "reactant_relaxed": reactant_ok,
            "product_relaxed": product_ok,
            "adatom_height_angstrom": height,
            "sites": ["fcc", "hcp"],
        },
    )
    top = slab.positions[:, 2] > (slab.positions[:, 2].max() - 0.5)
    adatom = reactant[-1]
    surface = reactant[:-1][top]
    surface_ids = np.flatnonzero(top)
    delta = mic_vectors(np.repeat(adatom[None, :], len(surface), axis=0), surface, cell)
    neighbor = int(surface_ids[int(np.argmin(np.linalg.norm(delta, axis=1)))])
    swapped = reactant.copy()
    swapped[-1], swapped[neighbor] = reactant[neighbor].copy(), reactant[-1].copy()
    exchange_product, exchange_energy, exchange_ok = relax(
        template,
        morse,
        swapped,
        fixed,
        PROTOCOL["endpoint_fmax_ev_per_angstrom"],
        PROTOCOL["endpoint_max_iterations"],
    )
    exchange = pack_system(
        "pt_exchange",
        "morse",
        True,
        numbers,
        cell,
        pbc,
        fixed,
        reactant,
        exchange_product,
        reactant_energy,
        exchange_energy,
        {
            "lattice_angstrom": lattice,
            "reactant_relaxed": reactant_ok,
            "product_relaxed": exchange_ok,
            "exchanged_index": neighbor,
        },
    )
    return hop, exchange


def build_vacancy(potential, lattice):
    import numpy as np
    from ase.build import bulk

    block = bulk("Al", "fcc", a=lattice, cubic=True).repeat((2, 2, 2))
    vacant = block.positions[0].copy()
    del block[0]
    numbers = block.numbers.copy()
    cell = block.cell.array.copy()
    pbc = np.array([True, True, True])
    reactant_guess = block.positions.copy()
    stacked_origin = np.repeat(vacant[None, :], len(reactant_guess), axis=0)
    delta = mic_vectors(stacked_origin, reactant_guess, cell)
    moving = int(np.argmin(np.linalg.norm(delta, axis=1)))
    distances = np.linalg.norm(delta, axis=1)
    order = np.argsort(-distances)
    pinned = None
    for index in order:
        if int(index) != moving:
            pinned = int(index)
            break
    fixed = np.zeros(len(numbers), dtype=np.uint8)
    fixed[pinned] = 1
    template = make_template(numbers, reactant_guess, cell, pbc)
    reactant, reactant_energy, reactant_ok = relax(
        template,
        potential,
        reactant_guess,
        fixed,
        PROTOCOL["endpoint_fmax_ev_per_angstrom"],
        PROTOCOL["endpoint_max_iterations"],
    )
    product_guess = reactant.copy()
    product_guess[moving] = vacant
    product, product_energy, product_ok = relax(
        template,
        potential,
        product_guess,
        fixed,
        PROTOCOL["endpoint_fmax_ev_per_angstrom"],
        PROTOCOL["endpoint_max_iterations"],
    )
    return pack_system(
        "al_vacancy",
        "eam_al",
        True,
        numbers,
        cell,
        pbc,
        fixed,
        reactant,
        product,
        reactant_energy,
        product_energy,
        {
            "lattice_angstrom": lattice,
            "reactant_relaxed": reactant_ok,
            "product_relaxed": product_ok,
            "moving_index": moving,
            "pinned_index": pinned,
        },
    )


def build_lj7(potential):
    import numpy as np

    rmin = 2.0 ** (1.0 / 6.0)
    radius = rmin / (2.0 * math.sin(math.pi / 5.0))
    height = math.sqrt(rmin * rmin - radius * radius)
    bipyramid = []
    for index in range(5):
        angle = 2.0 * math.pi * index / 5.0
        bipyramid.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
    bipyramid.append((0.0, 0.0, height))
    bipyramid.append((0.0, 0.0, -height))
    side = rmin / math.sqrt(2.0)
    octa = np.array(
        [
            [side, 0.0, 0.0],
            [-side, 0.0, 0.0],
            [0.0, side, 0.0],
            [0.0, -side, 0.0],
            [0.0, 0.0, side],
            [0.0, 0.0, -side],
        ],
        dtype=np.float64,
    )
    face = np.array([[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]])
    centroid = face.mean(axis=0)
    direction = centroid / np.linalg.norm(centroid)
    # |t d - v|^2 = rmin^2, using the first face vertex.
    vertex = face[0]
    coeff = float(np.dot(direction, vertex))
    # t^2 - 2 coeff t + |v|^2 - rmin^2 = 0
    quad = coeff * coeff - (float(np.dot(vertex, vertex)) - rmin * rmin)
    scale = coeff + math.sqrt(max(quad, 0.0))
    cap = scale * direction
    capped = np.vstack([octa, cap])
    numbers = np.ones(7, dtype=int)
    cell = np.eye(3) * 40.0
    pbc = np.array([False, False, False])
    fixed = np.zeros(7, dtype=np.uint8)
    template = make_template(numbers, bipyramid, cell, pbc)
    reactant, reactant_energy, reactant_ok = relax(
        template,
        potential,
        bipyramid,
        fixed,
        PROTOCOL["endpoint_fmax_ev_per_angstrom"],
        PROTOCOL["endpoint_max_iterations"],
    )
    product, product_energy, product_ok = relax(
        template,
        potential,
        capped,
        fixed,
        PROTOCOL["endpoint_fmax_ev_per_angstrom"],
        PROTOCOL["endpoint_max_iterations"],
    )
    product = match_atoms(reactant, product)
    return pack_system(
        "lj7",
        "ljcluster",
        False,
        numbers,
        cell,
        pbc,
        fixed,
        reactant,
        product,
        reactant_energy,
        product_energy,
        {"reactant_relaxed": reactant_ok, "product_relaxed": product_ok},
    )


def match_atoms(reactant, product):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    left = np.asarray(reactant, dtype=np.float64)
    right = np.asarray(product, dtype=np.float64)
    left_s = left - left.mean(axis=0)
    right_s = right - right.mean(axis=0)
    cost = np.linalg.norm(left_s[:, None, :] - right_s[None, :, :], axis=2)
    _rows, columns = linear_sum_assignment(cost)
    return right[columns]


def build_cu4_h(potential):
    import numpy as np

    side = 2.55
    copper = np.array(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side / 2.0, side * math.sqrt(3.0) / 2.0, 0.0],
            [side / 2.0, side * math.sqrt(3.0) / 6.0, side * math.sqrt(6.0) / 3.0],
        ],
        dtype=np.float64,
    )
    faces = ((0, 1, 2), (0, 1, 3))
    guesses = []
    for face in faces:
        triangle = copper[list(face)]
        centre = triangle.mean(axis=0)
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal = normal / np.linalg.norm(normal)
        other = [index for index in range(4) if index not in face][0]
        if float(np.dot(normal, copper[other] - centre)) > 0.0:
            normal = -normal
        guesses.append(np.vstack([copper, centre + normal]))
    numbers = np.array([29, 29, 29, 29, 1], dtype=int)
    cell = np.eye(3) * 40.0
    pbc = np.array([False, False, False])
    fixed = np.zeros(5, dtype=np.uint8)
    template = make_template(numbers, guesses[0], cell, pbc)
    reactant, reactant_energy, reactant_ok = relax(
        template,
        potential,
        guesses[0],
        fixed,
        PROTOCOL["endpoint_fmax_ev_per_angstrom"],
        PROTOCOL["endpoint_max_iterations"],
    )
    product, product_energy, product_ok = relax(
        template,
        potential,
        guesses[1],
        fixed,
        PROTOCOL["endpoint_fmax_ev_per_angstrom"],
        PROTOCOL["endpoint_max_iterations"],
    )
    return pack_system(
        "cu4_h",
        "cuh2",
        False,
        numbers,
        cell,
        pbc,
        fixed,
        reactant,
        product,
        reactant_energy,
        product_energy,
        {"reactant_relaxed": reactant_ok, "product_relaxed": product_ok, "cu_side_angstrom": side},
    )


def potential_of(name, rgpot):
    if name == "morse":
        return rgpot.MorsePot()
    if name == "eam_al":
        return rgpot.EAMAlPot()
    if name == "ljcluster":
        return rgpot.LJClusterPot()
    if name == "cuh2":
        return rgpot.CuH2Pot()
    raise RuntimeError(f"unknown potential {name}")


def attach_runtime(system):
    system = dict(system)
    system["template"] = make_template(
        system["numbers"], system["reactant"], system["cell"], system["pbc"]
    )
    system["reactant"] = __import__("numpy").asarray(system["reactant"], dtype=float)
    system["product"] = __import__("numpy").asarray(system["product"], dtype=float)
    system["fixed"] = __import__("numpy").asarray(system["fixed"], dtype=np_uint8())
    system["cell"] = __import__("numpy").asarray(system["cell"], dtype=float)
    return system


def np_uint8():
    import numpy as np

    return np.uint8


def reference_saddle(library, system, potential, meter):
    meter.reset()
    band = linear_band(
        system["reactant"], system["product"], PROTOCOL["reference_images"]
    )
    calculator = new_calculator(potential)
    outcome = run_rgsaddle_band(
        library,
        system,
        band,
        True,
        PROTOCOL["reference_fmax_ev_per_angstrom"],
        PROTOCOL["reference_max_iterations"],
        calculator,
    )
    energies = outcome["cached_energies"]
    if energies is None:
        energies = [
            probe(system["template"], potential, row, system["fixed"])[0]
            for row in outcome["band"]
        ]
    chosen = select_image(outcome["band"], energies, outcome["ci_index"], True)
    position = outcome["band"][chosen]
    calls = outcome["force_calls"]
    kernel = outcome["kernel_s"]
    energy, fmax = probe(system["template"], potential, position, system["fixed"])
    certificate = outcome["certificate"] or {
        "certified": 0,
        "index": None,
        "converged": 0,
    }
    usable = (
        outcome["status"] == "converged"
        and int(certificate.get("certified") or 0) == 1
        and int(certificate.get("index") or -1) == 1
    )
    return {
        "usable": usable,
        "status": outcome["status"],
        "certified": certificate.get("certified"),
        "index": certificate.get("index"),
        "certificate_converged": certificate.get("converged"),
        "energy": energy,
        "barrier": energy - float(system["reactant_energy"]),
        "fmax_free": fmax,
        "force_calls": calls,
        "kernel_s": kernel,
        "iteration": outcome["iteration"],
        "positions": position,
    }


def binding_check(pyec, system, potential):
    import numpy as np

    calculator = new_calculator(potential)
    atoms = system["template"].copy()
    atoms.positions = np.asarray(system["reactant"], dtype=np.float64)
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    params = eon_parameters(pyec, "dimer")
    matter = eon_matter(pyec, system, system["reactant"], calculator, params)
    other_energy = float(matter.potential_energy)
    raw_forces = np.asarray(matter.forces_raw, dtype=np.float64)
    masked = np.asarray(matter.forces, dtype=np.float64)
    if abs(other_energy - energy) > 1.0e-6:
        raise RuntimeError(
            f"{system['name']} eOn energy differs from the calculator by {other_energy - energy}"
        )
    if not np.allclose(raw_forces, forces, atol=1.0e-6, rtol=1.0e-6):
        raise RuntimeError(f"{system['name']} eOn forces differ from the calculator")
    fixed = np.asarray(system["fixed"], dtype=bool)
    if fixed.any() and not np.allclose(masked[fixed], 0.0, atol=1.0e-6):
        raise RuntimeError(f"{system['name']} eOn left a force on a fixed atom")
    return {
        "energy": energy,
        "max_force_difference": float(np.max(np.abs(raw_forces - forces))),
    }


def public_system(system):
    hidden = {"template"}
    return {key: plain(value) for key, value in system.items() if key not in hidden}


def system_digest(system):
    payload = public_system(system)
    payload.pop("reference", None)
    payload.pop("digest", None)
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def setup(args, library, rgpot, pyec):
    print("lattice Pt", flush=True)
    meter_probe = rgpot.MorsePot()
    pt_lattice = lattice_constant(meter_probe, 78)
    print("lattice Pt", pt_lattice, flush=True)
    hop, exchange = build_pt(meter_probe, pt_lattice)
    print("built Pt", flush=True)
    al = rgpot.EAMAlPot()
    print("lattice Al", flush=True)
    al_lattice = lattice_constant(al, 13)
    print("lattice Al", al_lattice, flush=True)
    vacancy = build_vacancy(al, al_lattice)
    print("built vacancy", flush=True)
    lj = rgpot.LJClusterPot()
    cluster = build_lj7(lj)
    print("built lj7", flush=True)
    copper = rgpot.CuH2Pot()
    hydride = build_cu4_h(copper)
    print("built cu4_h", flush=True)
    systems = [hop, exchange, vacancy, cluster, hydride]
    for system in systems:
        system["endpoint_rmsd"] = distinct(system)
        system["template"] = make_template(
            system["numbers"], system["reactant"], system["cell"], system["pbc"]
        )
        system["reactant"] = __import__("numpy").asarray(system["reactant"], dtype=float)
        system["product"] = __import__("numpy").asarray(system["product"], dtype=float)
        system["fixed"] = __import__("numpy").asarray(system["fixed"], dtype=np_uint8())
        system["cell"] = __import__("numpy").asarray(system["cell"], dtype=float)
    pots = {
        "morse": meter_probe,
        "eam_al": al,
        "ljcluster": lj,
        "cuh2": copper,
    }
    checks = {}
    for system in systems:
        checks[system["name"]] = binding_check(pyec, system, pots[system["potential"]])
        system["reference"] = reference_saddle(
            library, system, pots[system["potential"]], meter_for(pots[system["potential"]])
        )
        print(
            "reference",
            system["name"],
            system["reference"]["status"],
            "index",
            system["reference"]["index"],
            "barrier",
            system["reference"]["barrier"],
            flush=True,
        )
    payload = {
        "protocol": PROTOCOL,
        "shas": args.shas,
        "binding_check": checks,
        "systems": {system["name"]: public_system(system) for system in systems},
    }
    for name, system in payload["systems"].items():
        system["digest"] = system_digest(system)
    dump_json(os.path.join(args.out, "systems.json"), payload)
    with open(os.path.join(args.out, "systems.json"), encoding="utf-8") as handle:
        frozen = json.load(handle)
    for name in SYSTEM_ORDER:
        loaded = attach_runtime(frozen["systems"][name])
        loaded["name"] = name
        if system_digest(public_system(loaded)) != frozen["systems"][name]["digest"]:
            raise RuntimeError(f"{name} digest does not reload")
    print("setup wrote systems", flush=True)


def measure(args, library, rgpot, pyec):
    with open(args.reference, encoding="utf-8") as handle:
        frozen = json.load(handle)
    systems = []
    for name in SYSTEM_ORDER:
        system = attach_runtime(frozen["systems"][name])
        system["name"] = name
        if system_digest(public_system(system)) != frozen["systems"][name]["digest"]:
            raise RuntimeError(f"{name} digest does not match the frozen systems")
        systems.append(system)
    pots = {}
    for system in systems:
        if system["potential"] not in pots:
            pot = potential_of(system["potential"], rgpot)
            pots[system["potential"]] = pot
    jobs = []
    for system in systems:
        for images in PROTOCOL["images_including_endpoints"]:
            for method in NEB_METHODS:
                jobs.append(("neb", system, method, images))
        for method in MM_METHODS:
            jobs.append(("mm", system, method, None))
    runs = []
    recorded = versions(rgpot, pyec, library)
    for round_index in range(args.rounds):
        order = "forward" if round_index % 2 == 0 else "reversed"
        sequence = jobs if round_index % 2 == 0 else list(reversed(jobs))
        for kind, system, method, images in sequence:
            potential = pots[system["potential"]]
            meter = meter_for(potential)
            meter.reset()
            try:
                if kind == "neb":
                    outcome = execute_neb(
                        library, pyec, system, method, images, potential
                    )
                else:
                    outcome = execute_mm(
                        library, pyec, system, method, round_index, potential
                    )
                record = finish_record(
                    system,
                    method,
                    round_index,
                    order,
                    images,
                    None if kind == "neb" else round_index,
                    outcome,
                    potential,
                )
            except Exception as exc:
                calls, kernel = meter.snap()
                record = failed_record(
                    system,
                    method,
                    round_index,
                    order,
                    images,
                    None if kind == "neb" else round_index,
                    exc,
                    calls,
                    kernel,
                )
            runs.append(record)
            print(
                "run",
                round_index,
                system["name"],
                method,
                images if images is not None else "-",
                record["status"],
                record["success"],
                fmt(record["wall_s"]),
                record["force_calls"],
                flush=True,
            )
        summary, comparisons = summarize(runs)
        payload = coverage_record(
            args,
            recorded,
            {
                "complete": round_index + 1 == args.rounds,
                "systems": {name: frozen["systems"][name] for name in SYSTEM_ORDER},
                "summary": summary,
                "comparisons": comparisons,
                "runs": runs,
            },
        )
        dump_json(os.path.join(args.out, "results.json"), payload)
        write_table(os.path.join(args.out, "results.md"), payload)
    print("measure complete", len(runs), flush=True)


def coverage_record(args, recorded, body):
    recorded = dict(recorded)
    recorded.setdefault("eon_project", EON_PROJECT_VERSION)
    body = dict(body)
    body["protocol"] = PROTOCOL
    body["shas"] = args.shas
    body["rounds"] = args.rounds
    body["threads"] = args.threads
    body["cpu_model"] = host_cpu_model()
    body["versions"] = recorded
    return body


def versions(rgpot, pyec, library):
    import ase
    import numpy
    import scipy

    recorded = {
        "rgpot": getattr(rgpot, "__version__", None),
        "pyeonclient": getattr(pyec, "__version__", None),
        "rgsaddle_abi": library.abi,
        "ase": ase.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "eon_project": EON_PROJECT_VERSION,
    }
    for name in ("rustc", "gcc", "g++"):
        line = tool_line(name)
        if line:
            recorded[name] = line
    return recorded


def main(argv):
    args = parse_args(argv)
    args.out = os.path.abspath(args.out)
    args.rgsaddle_lib = os.path.abspath(args.rgsaddle_lib)
    args.libdir = os.path.abspath(args.libdir)
    args.pythonpath = [os.path.abspath(entry) for entry in args.pythonpath]
    if args.reference:
        args.reference = os.path.abspath(args.reference)
    pin(args)
    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)
    import rgpot
    import pyeonclient as pyec

    library = SaddleLibrary(args.rgsaddle_lib)
    bind_minimage(library)
    bind_calculator()
    recorded = versions(rgpot, pyec, library)
    if args.phase == "setup":
        setup(args, library, rgpot, pyec)
    else:
        measure(args, library, rgpot, pyec)
    with open(os.path.join(args.out, "versions.json"), "w", encoding="utf-8") as handle:
        json.dump(recorded, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
