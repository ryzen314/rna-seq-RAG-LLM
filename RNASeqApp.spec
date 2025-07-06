# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_all

# --- Project Configuration ---
spec_file_path = os.path.abspath(sys.argv[0])
project_root = os.path.dirname(spec_file_path)

app_name = 'RNASeqApp'

app_icon_path = os.path.join(project_root, 'RNASeqApp.icns') 
if not os.path.exists(app_icon_path):
    print(f"WARNING: Icon file not found at {app_icon_path}. Application will be built without an icon.")
    app_icon_path = None

# --- End Project Configuration ---

block_cipher = None

# Determine the correct path to site-packages in your active venv
site_packages_path = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
print(f"INFO: Determined site-packages path: {site_packages_path}")


# --- Analysis Phase ---
a = Analysis(
    ['main.py'],
    pathex=[project_root],
    binaries=[],
    datas=[
        # REMOVE THESE LINES if you don't want to bundle your local dev copies
        # (os.path.join(project_root, 'chrome_langchain_db'), 'chrome_langchain_db'),
        # (os.path.join(project_root, 'excel_files'), 'excel_files'),

        # KEEP THIS ONE, as it's the critical fix for langchain_core imports:
        (os.path.join(site_packages_path, 'langchain_core'), 'langchain_core'),
    ],
    hiddenimports=[
        'tkinter',
        'pydantic',
        'pydantic_core',
        # 'langchain_core', # Handled by direct copy
        'langchain_community',
        'langchain_ollama',
        'langchain_chroma',
        'langchain.chains.retrieval',
        'langchain.chains.combine_documents',
        # 'langchain.chains.Youtubeing', # Still verify/remove if not real
        'langchain.chains.conversational_retrieval',
        'langchain.chains.llm',
        'langchain.chains.retrieval_qa.base',
        'opentelemetry.context.contextvars_context',
        'pydantic.deprecated.decorator',
        'opentelemetry.sdk',
        'opentelemetry.instrumentation',
        'opentelemetry.instrumentation.langchain',
        'opentelemetry.instrumentation.openai',
        'importlib_metadata',
        'packaging',
        'setuptools',
        'langchain_core.callbacks.manager', # Keep for safety
        'langchain_core.callbacks.base',
        'langchain_core.tracers.base',
        'langchain_core.tracers.stdout',
        'langchain_core.runnables.base',
        'langchain_core.messages.base',
        'langchain_core.tools.base',
        'pandas._libs.tslibs.timestamps',
        'pandas._libs.interval',
        'openpyxl.worksheet._read_only',
        'openpyxl.worksheet.cell_row_dim',
        'openpyxl.styles',
        'lxml.etree', # Proactive for potential XML issues with openpyxl
        'lxml._elementpath', # Proactive for potential XML issues with openpyxl
        'xml.etree.ElementTree', # Standard XML library, sometimes needed implicitly
        'chromadb.telemetry.product.posthog', 
        'chromadb.telemetry',
        'chromadb.telemetry.product',
        'posthog',
        'chromadb.api.rust',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'matplotlib.tests', 'numpy.tests',
        'test', 'tests',
        'unittest',
        # Keep html, http, xml, as you are using openpyxl which might need some xml parsing
        # 'html',
        # 'http',
        # 'xml',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# --- Collect additional files from complex packages using collect_all ---
langchain_packages_to_collect = [
    'langchain',
    # 'langchain_core', # Removed - handled by explicit 'datas'
    'langchain_community',
    'langchain_ollama',
    'langchain_chroma',
    'langchain_text_splitters',
    'langchain_openai',
]

extra_binaries = []
extra_datas = []
extra_hiddenimports = []

for pkg in langchain_packages_to_collect:
    print(f"INFO: Collecting from package: {pkg}")
    collected_b, collected_d, collected_h = collect_all(pkg)
    extra_binaries.extend(collected_b)
    extra_datas.extend(collected_d)
    extra_hiddenimports.extend(collected_h)

a.binaries += extra_binaries
a.datas += extra_datas
a.hiddenimports += extra_hiddenimports

# --- CRITICAL FIX: ENSURE ALL ENTRIES IN a.binaries AND a.datas ARE 3-ELEMENT TUPLES ---
fixed_a_binaries = []
for item in a.binaries:
    if len(item) == 2:
        src, dest = item
        typecode = 'PYSOURCE' if src.lower().endswith(('.py', '.pyc', '.pyd')) else 'BINARY'
        fixed_a_binaries.append((src, dest, typecode))
    else:
        fixed_a_binaries.append(item)
a.binaries = fixed_a_binaries

fixed_a_datas = []
for item in a.datas:
    if len(item) == 2:
        src, dest = item
        typecode = 'DATA'
        if src.lower().endswith(('.py', '.pyc')):
             typecode = 'PYSOURCE'
        fixed_a_datas.append((src, dest, typecode))
    else:
        fixed_a_datas.append(item)
a.datas = fixed_a_datas

# --- END CRITICAL FIX ---


# --- PYZ Phase ---
pyz = PYZ(a.pure, a.zipped_data,
              cipher=block_cipher)

# --- EXE Phase ---
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name=app_name,
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True, # Keep True for now!
          disable_windowed_traceback=False,
          argv_emulation=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None,
          icon=app_icon_path
          )

# --- BUNDLE Phase ---
app = BUNDLE(exe,
             name=f'{app_name}.app',
             icon=app_icon_path,
             bundle_identifier="com.yourcompany.RNASeqApp",
             info_plist={
                 'CFBundleDisplayName': app_name,
                 'CFBundleShortVersionString': '1.0.0',
                 'CFBundleVersion': '1',
                 'NSHighResolutionCapable': True,
             }
            )
