from setuptools import setup,find_packages

config = {
    'include_package_data': True,
    'description': 'Multitask batch generation for training deeplearning models on CHIP-seq, CHIP-exo, CHIP-nexus, ATAC-seq, RNA-seq (or any other genomics assays that use highthroughput sequencing)',
    'download_url': 'https://github.com/kundajelab/mseqgen',
    'version': '0.1',
    'packages': ['mseqgen'],
    'setup_requires': [],
    'install_requires': ['numpy', 'pandas', 'scipy', 'deeptools', 'pyfaidx'],
    'name': 'mseqgen'
}

if __name__== '__main__':
    setup(**config)
