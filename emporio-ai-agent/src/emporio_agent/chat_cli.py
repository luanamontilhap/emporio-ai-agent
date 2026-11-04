"""Interactive chat with the LLM-backed agent: ``python -m emporio_agent chat [--trace]``."""
from __future__ import annotations

import argparse
import logging

from .agent import AgentResponse, build_agent
from .config import ConfigError, Settings

HELP = "Comandos: /novo (nova conversa), /trace (liga/desliga detalhes), /sair"


def _print_trace(response: AgentResponse) -> None:
    print(f"  [status={response.status}"
          + (f" motivo={response.handoff_reason}" if response.handoff_reason else "")
          + f" chamadas_llm={response.llm_calls}]")
    for call in response.tool_trace:
        outcome = "ok" if call.ok else f"erro: {call.error}"
        print(f"  [ferramenta] {call.name}({call.arguments}) -> {outcome} ({call.duration_ms} ms)")
    for violation in response.violations:
        print(f"  [validação] rascunho rejeitado: {violation.code} — {violation.detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m emporio_agent chat")
    parser.add_argument("--trace", action="store_true", help="mostra ferramentas chamadas e validações")
    parser.add_argument("--env-file", help="caminho do arquivo .env (padrão: .env na raiz do projeto)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    try:
        settings = Settings.from_env(dotenv_path=args.env_file)
    except ConfigError as exc:
        print(f"Configuração inválida: {exc}")
        return 2

    print("Carregando dados, manual de políticas e modelo de embeddings (1ª execução pode baixar o modelo)...")
    agent = build_agent(settings)
    session = agent.new_session()
    show_trace = args.trace
    print(f"Empório da Música — atendimento virtual (modelo: {settings.model} @ {settings.base_url})")
    print(HELP)

    while True:
        try:
            raw = input("\nVocê> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue
        command = raw.casefold()
        if command in {"/sair", "/exit", "exit"}:
            return 0
        if command == "/novo":
            session = agent.new_session()
            print("Nova conversa iniciada.")
            continue
        if command == "/trace":
            show_trace = not show_trace
            print(f"Detalhes {'ligados' if show_trace else 'desligados'}.")
            continue

        response = agent.ask(raw, session)
        print(f"\nAtendente> {response.text}")
        if response.citations:
            pages = ", ".join(sorted({f"p. {c['page']}" for c in response.citations}, key=lambda s: int(s[3:])))
            print(f"  (trechos do manual consultados: {pages})")
        if show_trace:
            _print_trace(response)


if __name__ == "__main__":
    raise SystemExit(main())
