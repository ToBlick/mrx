Installation
============

This page provides general information on installation.

Requirements
------------

``mrx`` is a python package focused on stellarator optimization
and requires python version 3.9 or higher.  ``mrx`` also requires
some mandatory python packages, listed in
``requirements.txt``.
These packages are all installed automatically when you install using
``pip`` or another python package manager such as ``conda``, as
discussed below.  You can also manually install these python packages
using ``pip`` or ``conda``, e.g. with ``pip install -r requirements.txt``.

Installing MRX
--------------

You can install ``mrx`` in editable mode with ``pip`` by navigating to the root directory of the repository and running:

.. code-block:: bash

    pip install -e .

Optional Packages
-----------------

- For GPU support install jax with cuda support:
    ``pip install -U "jax[cuda12]"``

- For building the documentation, navigate to docs/ and run:
    ``pip install -r requirements.txt``
    ``conda install -c conda-forge doxygen pandoc``
    ``make html``

Post-Installation
-----------------

If the installation is successful, ``mrx`` will be added to your
python environment. You should now be able to import the module from
python::

  >>> import mrx

