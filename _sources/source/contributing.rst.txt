.. _contributing guidelines:


Contribution guidelines for compIAM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Collaborators in the MTG will be able to contribute directly to the comIAM repo. It is important that you have write 
access to the compiam repo (`speak to Genís Plaja <mailto:genis.plaja@upf.edu>`_ if this is not the case, requesting 
access to push into the repo). 

Please contact `Genís Plaja <mailto:genis.plaja@upf.edu>`_ or `Thomas Nuttall <mailto:thomas.nuttall@upf.edu>`_ for any 
questions or doubts.


How can I contribute?
---------------------

There are several ways to contribute to compiam. We will now try to list them all:

* Integrating a **new tool/model** (or an upgraded version of an existing model)
* Integrating **access to a new dataset** (or to a dataset that has already been around for some time but the access to it is not canonical and standardized)
* **Fixing a bug**
* Improving the library itself with **new core features** and **structure improvements**
* Obviously, **using it** 🙂


Accepting a feature
-------------------
We recommend that, if you want to contribute into compiam through one of the candidates in the list above, 
you let us know through the compIAM Slack workspace. If you are still not signed up in the compIAM Slack workspace,
please email us and we will create a user for you. You can also interact with us by opening an issue in 
`GitHub issues <https://github.com/MTG/compIAM/issues>`_ of our repo, and we can even discuss in person/online call. 
Feel free to get in touch with us through email to organize an online meeting if necessary.

.. important::
    We will try to also actively propose new features to include in  compiam, so stay tuned in case you are interested in
    working with some of these!

In the following sections, we provide a walkthrough of how one can contribute to compiam through our repository in GitHub.


Development step 1: setting up the environment
----------------------------------------------

It is recommended that you set up a virtual environment and install compiam to this environment before beginning any development. Let's see how this is done:

* You can create a virtual environment using ``conda`` or ``pyenv``

* Clone the repository to your machine: ``git clone https://github.com/MTG/compIAM.git``

* Install the compiam package in the virtual environment from source code: ``pip install -e .``

* During the development, you will need additional dependencies for testing and documenting. Install them as follows:
    
        * ``pip install -e ."[tests]"`` to install the testing dependencies.
    
        * ``pip install -e ."[docs]"`` to install the dependencies for the online documentation.


You are now ready to go!


Development step 2: write your feature
--------------------------------------

#. Clone the repository to your machine: 

    * Open a terminal and ``cd`` into the ``compIAM`` main folder

    * Create a branch from master named ``yn-featname`` where ``yn`` is replaced with your initials and ``featname`` with a short name of the feature you will implement

    * This can be done through the terminal with ``git checkout -b yn-featname``, which will also move you to the branch.

    .. tip::
        Additional information about branches in git repositories can be `found in this link <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_.
        Make sure you understand how branching in GitHub works not only to properly contribute to  compiam  but also in other projects in the future!

#. Implement new features: 

    * 📓 **Here are some best practices to follow when developing your feature:**

        * Try to adhere to `PEP-8 style guidelines <https://peps.python.org/pep-0008/>`_.

        * Try and keep development strictly to the feature you are implementing (don't mix with other tasks). If other problems arise while developing your new feature or fix, please `open an issue <https://github.com/MTG/compIAM/issues>`_.

        * Follow the compiam `naming/structure conventionss <https://docs.google.com/document/d/13WcMtii0gLm_ocU3MTqFG59lbMYBduJEgypLlsbLDGQ/edit#>`_.
        
        * Functions should include docstrings in the Sphinx format. You can see examples in other modules in compiam. Documentation is very important for users of compiam, so please spend time on it.
        
    * **Commit regularly with relevant commit messages using:**

        * After making a substantial change in the code in your branch, it might be a good moment to commit the changes.

        * Include files to your commit: ``git add .`` (``git add .`` will include all files in the repository in the commit. To include only some changes you made and commit other ones later, you can specify ``git add file_1.py file_2.py``)

        * Create and name a commit: ``git commit -m “short action name”``

        * Push changes online: ``git push origin yn-featname``

    * **If you are adding new tools:**

        * Please add the proper citation of the work you are integrating, in order for users to properly cite such work in case they use it for their own research.

    * **If you are adding new models:**

        * If you add a model that has pre-trained weights available, you may want to include these in compIAM as well. Then, your model will be loaded using the ``.load_model()`` wrapper. Here are some TODOs to add a pre-trained model to compIAM:

            * Write a model wrapper that has all util functions to download and load the weights, build the model, and run prediction/extraction. You may want to use one of the existing model classes as en example (e.g., FTANetCarnatic). **We will publish a template soon :)**

            .. note::
                Do not hesitate to get in touch if you have any doubt/question on integrating the model to compIAM. We will be happy to 
                assist you as soon as possible :)


            * Add your model to ``models_dict`` in compiam/data.py, adding all details about your model. There are instructions on the file itself.

            * Upload your weights to Zenodo. You will need to download link and checksum for the model information in ``models_dict``. You should upload a compressed folder containing the weights and additional files related to the pre-trained model (not code!).

            * Make sure your model is properly initialized, built, and that the weights are properly downloaded and loaded. Using the example functions from another model in compIAM, all these actions can be executed.

        .. note::
            Since we support Machine and Deep Learning models, we are open to including dependencies such as torch, tensorflow,
            and others. These, however, are included as “optional dependencies”, so  you only install those if necessary. Check out
            examples such as ``Melodia`` or ``FTANetCarnatic`` to learn how that works! 

    * **If you are adding new datasets:**

        * To host dataset loaders in compiam, we inherit from `mirdata <https://github.com/mir-dataset-loaders/mirdata>`_. In other words, to include a dataset in compiam, the dataset needs first to be present in ``mirdata``. 

        * Including dataset loaders in mirdata is very beneficial for the community. Write a data loader for your dataset there!!

        * Once you get your data loader in mirdata, open an issue in compiam letting us know and we can integrate it to compiam as well.

        .. note::
            ``mirdata`` is now very mature, including several maintainers and contributors, and therefore, it is very much contributor
            friendly, **as we intend to be!** 



Development step 3: test your feature
-------------------------------------

When finished, if appropriate, add unit tests to the testing framework in ``tests/`` (using the Python testing package `pytest <https://docs.pytest.org/en/7.1.x/>`_).

* Typical tests that you should write include checking that the tool is loaded with no unexpected bugs and that if running the tool on top of an example track, the output is as expected.


.. note::
    You may include one or more short audio examples in ``tests/resources/`` to test that the output of your tool is correct.


* Run ``pytest ./tests/``

* You can run specific tests depending on the dependencies of your work. For instance, if you are working on a TensorFlow model, you should run:

    * ``pytest ./tests/ --tensorflow``

    * ``pytest ./tests/ --full-ml``

If you miss a particular test don't worry. We automatically run the tests when you create the PR.


Development step 4: creating a pull request (PR)
------------------------------------------------

**If all tests pass with the feature that you have implemented create a pull request (PR) to master and request at least one fellow collaborator to review.**

* Relevant resources are `found here <https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/working-with-your-remote-repository-on-github-or-github-enterprise/creating-an-issue-or-pull-request>`_.

* To create a PR you can go to the repository of compiam on the GitHub website, go to your branch, and create a PR from there. 

* Essentially, in a PR you are requesting the repository maintainers to merge your code updates to the master branch so that users can use your implementations.

**A compiam maintainer will comment on your code and potentially request changes:**

* If one of their comments requires a change, you must either make that change or respond by explaining why you think it is not necessary.

* Once you and the reviewer have reached an agreement on the code they will tell you it is OK to merge. 

**Once you have received the OK from maintainers, you can merge the pull request to master!** You can now delete the feature branch ``yn-featname``.


Post-development
----------------

Cool!! Your tool is now merged in compiam, and will be published in ``pypi`` in the next release.
Make sure you also take advantage of this feature, spread the word, and please keep being part of the compiam community. 

Some future PRs might involve your tool (change of inner structure of the library, updating dependencies, etc…).
Also, there might be users finding bugs or having issues/difficulties using your tool. Please make sure you check
regularly the Issues/Pull Requests tab of compiam, in case there is a user requesting your attention. Thank you
for your cooperation!