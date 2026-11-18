"""Offline tests: per-tier key names, and installing a provider's package on first use.

No network: the installer is replaced by a stub that records the command.
"""

import os

import steerable
from steerable import ModelRouterMiddleware, ensure_provider, provider_package, resolve_model

os.environ.update(WORK_ANTHROPIC_KEY="sk-ant-work-test", PERSONAL_ANTHROPIC_KEY="sk-ant-personal-test",
                  SECOND_OPENROUTER_KEY="sk-or-second-test", OPENROUTER_API_KEY="sk-or-default-test")


def secret(model):
    for field in ("anthropic_api_key", "openai_api_key", "google_api_key"):
        if getattr(model, field, None) is not None:
            return getattr(model, field).get_secret_value()


# --- per-tier key names ------------------------------------------------------

main = resolve_model("anthropic:claude-sonnet-5-5", "WORK_ANTHROPIC_KEY")
r = ModelRouterMiddleware("anthropic:claude-haiku-5-5", ["openrouter:a/b:free"], check_tools=False,
                          cheap_key="PERSONAL_ANTHROPIC_KEY", free_key="SECOND_OPENROUTER_KEY")
assert secret(main) == "sk-ant-work-test" and secret(r.chain[1]) == "sk-ant-personal-test"
assert secret(r.chain[2]) == "sk-or-second-test"
assert secret(resolve_model("openrouter:a/b:free")) == "sk-or-default-test", "no key name: the provider's usual variable"
print("PASS two tiers on the same provider use different keys, each named by a .env variable")

try:
    resolve_model("anthropic:claude-sonnet-5-5", "TYPO_KEY")
    raise AssertionError("a missing key name must fail")
except ValueError as e:
    assert str(e) == "TYPO_KEY is not set; add it to .env", e
print("PASS a key name that isn't in .env fails up front, naming the variable (never the key)")

# --- installing a provider's package on first use ----------------------------

assert provider_package("mistralai") == "langchain-mistralai"
assert provider_package("google_vertexai") == "langchain-google-vertexai"
assert provider_package("openrouter") == "langchain-openai" and provider_package("made-up") is None

ran = []
ensure_provider("anthropic", run=lambda cmd, check: ran.append(cmd))
assert ran == [], "an installed provider installs nothing"

real_find = steerable.importlib.util.find_spec
steerable.importlib.util.find_spec = lambda name: None if name == "langchain_mistralai" else real_find(name)
try:
    ensure_provider("mistralai", run=lambda cmd, check: ran.append(cmd))
    assert ran and ran[0][-1] == "langchain-mistralai" and steerable.sys.executable in ran[0], ran
    ensure_provider("made-up", run=lambda cmd, check: ran.append(cmd))
    assert len(ran) == 1, "only providers in LangChain's own table are ever installed"
    os.environ["HARNESS_AUTO_INSTALL"] = "0"
    try:
        ensure_provider("mistralai", run=lambda cmd, check: ran.append(cmd))
        raise AssertionError("auto-install off must not install")
    except ImportError as e:
        assert "pip install langchain-mistralai" in str(e)
finally:
    steerable.importlib.util.find_spec = real_find
    os.environ.pop("HARNESS_AUTO_INSTALL", None)
print("PASS a missing provider package is installed into this venv on first use:", " ".join(ran[0][-4:]))

print("ALL-KEY-TESTS-OK")
