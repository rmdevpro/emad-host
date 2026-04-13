#!/bin/bash
# pMAD Template — Entrypoint script
# Installs AE and TE packages from the configured source before starting
# the application server. Package versions are specified in config.yml.

set -e

# ERQ-005 §2.4: umask 000 for world-writable file creation
umask 000

# Defense-in-depth: data dirs should exist from git clone (.gitkeep),
# but guard against edge cases where bind mount has wrong permissions.
mkdir -p /data/downloads 2>/dev/null || true

CONFIG_FILE="${CONFIG_PATH:-/config/config.yml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found at $CONFIG_FILE — cannot install packages"
    exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
fi

# Extract package config from config.yml using Python
eval $(python3 -c "
import yaml
try:
    with open('$CONFIG_FILE') as f:
        cfg = yaml.safe_load(f)
    pkgs = cfg.get('packages', {})
    print(f'PKG_SOURCE={pkgs.get(\"source\", \"pypi\")}')
    print(f'PKG_LOCAL_PATH={pkgs.get(\"local_path\", \"/app/packages\")}')
    print(f'PKG_DEVPI_URL={pkgs.get(\"devpi_url\", \"\")}')
    print(f'PKG_AE={pkgs.get(\"ae\", \"\")}')
    print(f'PKG_TE={pkgs.get(\"te\", \"\")}')
except Exception as e:
    print('PKG_SOURCE=pypi')
    print('PKG_LOCAL_PATH=/app/packages')
    print('PKG_DEVPI_URL=')
    print('PKG_AE=')
    print('PKG_TE=')
" 2>/dev/null)

echo "Package source: $PKG_SOURCE"
echo "AE package: ${PKG_AE:-none}"
echo "TE package: ${PKG_TE:-none}"

# ── Install AE and TE packages ────────────────────────────────────────
# Packages are installed with --no-deps because all framework dependencies
# (langgraph, langchain, etc.) are already in the image from requirements.txt.
# Only the AE/TE code itself is installed.

install_package() {
    local pkg_spec="$1"
    if [ -z "$pkg_spec" ]; then return; fi

    # Extract package name (without version) for local path
    local pkg_name="${pkg_spec%%==*}"

    case "$PKG_SOURCE" in
        local)
            local src_dir="$PKG_LOCAL_PATH/$pkg_name"
            if [ -d "$src_dir" ]; then
                # RB-34: packages mount is :ro — copy to /tmp for install
                local tmp_dir="/tmp/pkg-install-$pkg_name"
                rm -rf "$tmp_dir"
                cp -r "$src_dir" "$tmp_dir"
                echo "Installing $pkg_name from local: $src_dir"
                pip install --user --no-deps --no-cache-dir --force-reinstall "$tmp_dir"
                rm -rf "$tmp_dir"
            else
                echo "WARNING: Local source not found: $src_dir"
            fi
            ;;
        devpi)
            if [ -n "$PKG_DEVPI_URL" ]; then
                echo "Installing $pkg_spec from devpi"
                pip install --user --no-deps --no-cache-dir --index-url "$PKG_DEVPI_URL" "$pkg_spec"
            else
                echo "WARNING: devpi_url not set, cannot install $pkg_spec"
            fi
            ;;
        pypi|*)
            echo "Installing $pkg_spec from PyPI"
            pip install --user --no-deps --no-cache-dir "$pkg_spec"
            ;;
    esac
}

install_package "$PKG_AE"
install_package "$PKG_TE"

# ── Install eMAD packages ─────────────────────────────────────────────
# eMAD packages have platform-specific deps (google-api-python-client,
# msal, etc.) NOT in the base image, so they install WITH dependencies.

install_emad_package() {
    local pkg_spec="$1"
    if [ -z "$pkg_spec" ]; then return; fi

    case "$PKG_SOURCE" in
        devpi)
            if [ -n "$PKG_DEVPI_URL" ]; then
                echo "Installing eMAD $pkg_spec from devpi"
                pip install --user --no-cache-dir --index-url "$PKG_DEVPI_URL" "$pkg_spec"
            else
                echo "WARNING: devpi_url not set, cannot install $pkg_spec"
            fi
            ;;
        pypi|*)
            echo "Installing eMAD $pkg_spec from PyPI"
            pip install --user --no-cache-dir "$pkg_spec"
            ;;
    esac
}

# Install eMAD packages from /emads/ configs (runbook-based eMADs)
if [ -d "/emads" ]; then
    EMAD_TE_INSTALLED=""
    for emad_dir in /emads/*/; do
        if [ -f "${emad_dir}config.json" ]; then
            emad_name=$(basename "$emad_dir")
            echo "eMAD found: $emad_name"
            if [ -z "$EMAD_TE_INSTALLED" ]; then
                install_emad_package "runbook-emad-te"
                EMAD_TE_INSTALLED="yes"
            fi
        fi
    done
fi

# Install eMAD packages registered in the database
python3 -c "
import asyncio, asyncpg, os
async def main():
    dsn = 'postgresql://{}:{}@{}:{}/{}'.format(
        os.environ.get('POSTGRES_USER', 'emad_host'),
        os.environ.get('POSTGRES_PASSWORD', ''),
        os.environ.get('POSTGRES_HOST', 'emad-host-postgres'),
        os.environ.get('POSTGRES_PORT', '5432'),
        os.environ.get('POSTGRES_DB', 'emad_host'),
    )
    try:
        conn = await asyncpg.connect(dsn, timeout=5)
        rows = await conn.fetch(
            'SELECT DISTINCT package_name, installed_version FROM emad_packages WHERE status = \$1',
            'active',
        )
        await conn.close()
        for row in rows:
            pkg = row['package_name']
            ver = row['installed_version']
            if ver and ver != 'unknown':
                print(f'{pkg}=={ver}')
            else:
                print(pkg)
    except Exception as e:
        import sys
        print(f'WARNING: Could not read emad_packages: {e}', file=sys.stderr)
asyncio.run(main())
" 2>/dev/null | while read pkg_spec; do
    install_emad_package "$pkg_spec"
done

# Start the application
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
