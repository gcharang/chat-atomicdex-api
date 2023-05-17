import weaviate
import json

w = weaviate.Client("http://weaviate:8080")

result = w.backup.create(
    backup_id='my-very-first-backup',
    backend='filesystem',
)

print(json.dumps(result, indent=4))