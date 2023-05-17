import weaviate

w = weaviate.Client("http://weaviate:8080")

result = w.backup.get_create_status(
  backup_id='my-very-first-backup',
  backend='filesystem',
)

print(result)