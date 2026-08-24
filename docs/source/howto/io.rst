

Load molecules through readcon
------------------------------

Every molecular frame in this stack is a CON frame.

.. table::

    +------------------------------------------------------------+---------------------------------------------------+
    | Crate                                                      | Role                                              |
    +============================================================+===================================================+
    | `readcon-core <https://github.com/lode-org/readcon-core>`_ | CON reader / writer, C ABI, spec v2/v3            |
    +------------------------------------------------------------+---------------------------------------------------+
    | readcon-chemfiles (``chemfiles`` feature on readcon-core)  | Foreign trajectories (XYZ, PDB, ...) into CON     |
    +------------------------------------------------------------+---------------------------------------------------+
    | `readcon-db <https://github.com/lode-org/readcon-db>`_     | mmap corpus for traces and restarts               |
    +------------------------------------------------------------+---------------------------------------------------+
    | `linkcell <https://github.com/d-SEAMS/linkcell>`_          | Periodic MIC and linked-cell k-nearest            |
    +------------------------------------------------------------+---------------------------------------------------+
    | `vesin <https://github.com/Luthaf/vesin>`_                 | Cutoff neighbour lists, when a surface needs them |
    +------------------------------------------------------------+---------------------------------------------------+

rgsaddle does not parse CON itself. Enable ``--features readcon`` and
call ``frame_from_con`` to get a 3N Cartesian, masses, and a
``linkcell::Cell``. Enable ``chemfiles`` for foreign formats through
readcon-core. Enable ``readcon-db`` when the host persists an IRC
or dimer trace. Enable ``vesin`` only if the potential needs a
cutoff pair list.

Minimum-image differences on the band go through
``wrap_difference`` (linkcell). They do not go through a private
3x3 box.
