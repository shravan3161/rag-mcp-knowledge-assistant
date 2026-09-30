from retrieval_tools import (
    search_troubleshooting,
    search_manual,
    search_warranty
)


query = "My printer is making a strange noise."


print("\n===== TROUBLESHOOTING =====")

results = search_troubleshooting(query)

for result in results:
    print(result.page_content)
    print()


print("\n===== MANUAL =====")

results = search_manual(query)

for result in results:
    print(result.page_content)
    print()


print("\n===== WARRANTY =====")

results = search_warranty(query)

for result in results:
    print(result.page_content)
    print()