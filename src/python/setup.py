from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.calsum",
    package_names=['zensols', 'resources'],
    # package_data={'': ['*.html', '*.js', '*.css', '*.map', '*.svg']},
    package_data={'': ['*.conf', '*.json', '*.yml']},
    description='Summarize text using Component ALignment Abstract Meaning Representation (CALAMR) alignment.',
    user='plandes',
    project='calsum',
    keywords=['tooling'],
    # has_entry_points=False,
).setup()
