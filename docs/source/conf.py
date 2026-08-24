#!/usr/bin/env python3

import os

project = "rgsaddle"
copyright = "2026--present, Rohit Goswami"
author = "Rohit Goswami"
html_logo = "_static/rgsaddle-notext-light.webp"
html_favicon = "_static/rgsaddle-icon.webp"

extensions = [
    "sphinxcontrib.bibtex",
    "myst_parser",
    "sphinx.ext.intersphinx",
]
try:
    import sphinxcontrib_rust  # noqa: F401
    import sphinx_rustdoc_postprocess  # noqa: F401

    extensions.extend(["sphinxcontrib_rust", "sphinx_rustdoc_postprocess"])
except ImportError:
    pass

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_timeout = 5

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "rgpot": ("https://rgpot.rgoswami.me", None),
    "eon": ("https://eondocs.org", None),
}

rust_crates = {
    "rgsaddle": os.path.abspath("../../"),
}
rust_doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crates")
rust_rustdoc_fmt = "rst"
rust_generate_mode = "always"

rustdoc_postprocess_toctree_target = "reference/rust-api.rst"
rustdoc_postprocess_toctree_rst = """
Rust API
--------

.. toctree::
   :maxdepth: 2

   ../crates/rgsaddle/lib
"""

html_theme = "shibuya"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/OmniPotentRPC/rgsaddle",
    "accent_color": "indigo",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    "light_logo": "_static/rgsaddle-notext-light.webp",
    "dark_logo": "_static/rgsaddle-notext-dark.webp",
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "rgmin",
                    "url": "https://github.com/OmniPotentRPC/rgmin",
                    "summary": "Local steppers and Riemannian manifolds",
                },
                {
                    "title": "rgpot",
                    "url": "https://rgpot.rgoswami.me",
                    "summary": "Potential-energy library and RPC server",
                },
                {
                    "title": "eOn",
                    "url": "https://eondocs.org",
                    "summary": "Saddle-point search on potential energy surfaces",
                },
                {
                    "title": "gpr_optim",
                    "url": "https://github.com/TheochemUI/gpr_optim",
                    "summary": "GP-NEB and IRCDriver host",
                },
            ],
        },
    ],
}

html_context = {
    "source_type": "github",
    "source_user": "OmniPotentRPC",
    "source_repo": "rgsaddle",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}

html_css_files = [
    "custom.css",
]

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
