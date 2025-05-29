Generating the Documentation
============================

Generate the documentation for the MRX package.

The documentation for the MRX package is generated automatically
from the main branch and can be viewed at https://mrx.readthedocs.io/en/latest/

To generate the documentation locally, e.g. if you are making 
changes to the documentation and want to check it before pushing 
changes to the main branch, follow the steps below.

1. Install the documentation dependencies:

.. code-block:: bash

    pip install -r docs/requirements.txt
    conda install -c conda-forge doxygen

2. Generate the documentation:

.. code-block:: bash

    cd docs 
    make html

3. The documentation will be generated in the `docs/build/html` directory.

4. To view the documentation, open the `index.html` file in your browser.

5. To rerun after changes, run:

.. code-block:: bash

    cd docs 
    make clean
    make html