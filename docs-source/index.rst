.. _rethon-docs-label:

rethon
======

A Python package for modeling the method of reflective equilibrium based on |bbb2021|.

Installation
------------

With :code:`pip` (👉 |rethon_pypi|)

.. code-block:: bash

    pip install rethon

From the source code:


You can install the package locally, by

* first git-cloning the repository:

  * :code:`git clone git@github.com:re-models/rethon.git`

* and then installing the package by running ':code:`pip install -e .`' from the local directory of
  the package (e.g. :code:`local-path-to-repository/rethon`) that contains the setup file :code:`setup.py`.
  (The :code:`-e`-option will install the package in the editable mode, allowing you to
  change the source code.)

.. note:: The package requires a python version >= 3.8 and depends on the
    |theodias| package, which will be installed automatically by pip.



Documentation
-------------


The :ref:`tutorials <rethon-tutorials-label>` provide step-by-step instructions of using
the :code:`rethon` package. Further details can be found in the :ref:`API documentation <rethon-api-docs-label>`.

.. toctree::
    :hidden:

    Tutorials <tutorials/rethon-tutorials>
    API documentation<api-docs/api>

Logging
-------

:code:`rethon` logs via a logger named 'rethon' and configures 3 loggers ('tau','rethon' and 'dd') as
specified in the configuration file code:`rethon.config.logging-config.json`.  Both 'tau' and 'rethon'
output their logs up to the DEBUG-Level to :code:`sys.stdout`, :code:`sys.stderr` and to the file
'rethon.log'. The 'dd'-logger uses the same output channels but does only log beginning from level ERROR.


If you want to customize the logging, you can remove the loggers' handler and specify you own
handlers (see, e.g., `<https://docs.python.org/3/howto/logging-cookbook.html>`_).


Licence
-------

|licence|

References
----------

.. |bbb2021| raw:: html

   <a href="https://doi.org/10.3998/ergo.1152" target="_blank">Beisbart, Betz and Brun (2021)</a>

.. |theodias| raw:: html

   <a href="https://github.com/re-models/theodias" target="_blank">theodias</a>

.. |rethon_pypi| raw:: html

   <a href="https://pypi.org/project/rethon/" target="_blank">https://pypi.org/project/rethon/</a>


.. |licence| raw:: html

   <a href="https://github.com/re-models/rethon?tab=MIT-1-ov-file#readme" target="_blank">MIT Licence</a>

