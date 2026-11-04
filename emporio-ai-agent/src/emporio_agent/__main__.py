import json
import sys
from pathlib import Path
from .data_loader import StoreData
from .tools import StoreTools


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "chat":  # LLM-backed agent (see chat_cli.py)
        from .chat_cli import main as chat_main
        raise SystemExit(chat_main(sys.argv[2:]))
    project_root = Path(__file__).resolve().parents[2]
    data = StoreData(project_root / "data")
    tools = StoreTools(data)
    print("Empório da Música — consulta local de ferramentas")
    policy_rag = None
    print("Comandos: products, product <id>, price <id>, order <id> <telefone>, info, diagnostics, policy <pergunta>, exit")
    print("Para conversar com o agente (LLM): python -m emporio_agent chat")
    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        parts = raw.split()
        command = parts[0].casefold()
        try:
            if command == "exit":
                break
            elif command == "products":
                result = tools.search_products(available_only=True)
            elif command == "product" and len(parts) == 2:
                result = tools.get_product_details(int(parts[1]))
            elif command == "price" and len(parts) == 2:
                result = tools.get_product_price(int(parts[1]))
            elif command == "order" and len(parts) >= 3:
                result = tools.get_order_details(int(parts[1]), " ".join(parts[2:]))
            elif command == "info":
                result = tools.get_store_information()
            elif command == "diagnostics":
                result = data.diagnostics()
            elif command == "policy" and len(parts) >= 2:
                if policy_rag is None:
                    from .policy_rag import PolicyRAG
                    print("Loading policy retriever and embedding model (first run may download the model)...")
                    policy_rag = PolicyRAG(project_root / "documents" / "politicas_da_loja.pdf")
                result = policy_rag.retrieve(" ".join(parts[1:]))
            else:
                print("Invalid command. See the command list.")
                continue
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        except (ValueError, TypeError) as exc:
            print(f"Invalid input: {exc}")


if __name__ == "__main__":
    main()
