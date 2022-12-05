import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "FateZ",
    version = "v0.0.1-alpha1",
    author = "JackSSk",
    author_email = "gyu17@alumni.jh.edu",
    description = "Ummmmmm, later!",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    # url = "https://github.com/JackSSK/Ageas",
    # project_urls = {"Bug Tracker": "https://github.com/JackSSK/Ageas/issues",},
    packages = setuptools.find_packages(),
    # package_data = {'': [
    #     'data/config/*',
    #     'data/human/*.txt',
    #     'data/human/bioGRiD.stratified.js.gz',
    #     'data/human/gtrd_promoter-1000.js.gz',
    #     # 'data/human/gtrd_whole_genes.js.gz',
    #     'data/human/uniprot_idmapping.stratified.js.gz',
    #     'data/mouse/*.txt',
    #     'data/mouse/bioGRiD.stratified.js.gz',
    #     'data/mouse/gtrd_promoter-1000.js.gz',
    #     # 'data/mouse/gtrd_whole_genes.js.gz',
    #     'data/mouse/uniprot_idmapping.stratified.js.gz',
    # ]},
    classifiers = [
                    'Programming Language :: Python :: 3',
                    'License :: OSI Approved :: MIT License',
                    'Operating System :: OS Independent',
                    ],
    python_requires = ">=3.6",
    include_package_data = True,
    install_requires = [
        'shap >= 0.40.0',
        'numpy >= 1.19.5',
        'scipy >= 1.8.0',
        'torch >= 1.11.0',
        'pandas >= 1.4.2',
        'netgraph >= 4.4.1',
        'xgboost >= 1.6.0',
        'networkx >= 2.8.1',
        'matplotlib >= 3.5.2',
        'scikit-learn >= 1.0.2',
    ],
)
