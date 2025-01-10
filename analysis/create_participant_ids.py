import uuid

def generate_participant_id(recruitment_prefix):
    random_part = str(uuid.uuid4().hex)[:8]  # Generate random portion of the ID
    participant_id = f"{recruitment_prefix}-{random_part}"
    return participant_id

n = 30

for i in range(n):
    print(generate_participant_id("AGE"))

