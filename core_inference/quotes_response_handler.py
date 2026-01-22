from core_inference.repository import Repository


class QuotesResponseHandler:
    def __init__(self, repository: Repository):
        self.repository = repository

    async def handle(self, data):
        self.repository.update_quote(data)

        
