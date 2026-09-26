

Architecture
------------

.. code:: dot

    digraph rgsaddle {
    graph [fontname="Jost", fontsize=12, rankdir=LR];
    node [fontname="Jost", fontsize=10, style=filled, fillcolor=white, color="#004D40"];
    edge [color="#004D40"];

    host [label="host\ngpr_optim / eOn"];
    band [label="BandSession"];
    mm [label="MinModeSession"];
    irc [label="IrcSession"];
    rgmin [label="rgmin Solver\nIrcTrust / LowestMode"];

    host -> band;
    host -> mm;
    host -> irc;
    band -> rgmin;
    mm -> rgmin;
    irc -> rgmin;
    }

``rgsaddle`` never implements a stepper. Every accepted move is one
rgmin ``Solver::step``, or an ``IrcTrust`` projection of a trial
increment. ``reset`` is the only model-update seam.
