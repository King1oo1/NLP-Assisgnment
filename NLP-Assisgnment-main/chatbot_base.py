"""
Base Chatbot class
Create a new file for your chatbot that inherits from this class
"""

class ChatbotBase:
    def __init__(self, name="Chatbot"):
        self.name = name
        self.conversation_is_active = True

    def greeting(self):
        print(f'Hello I am {self.name}')

    def farewell(self):
        print('Goodbye!')
    
    def conversation_is_active(self):
        return self.conversation_is_active

    def receive_input(self):
        user_input = input()
        return user_input

    def process_input(self, user_input):
        raise NotImplementedError('process_input() not implemented in base Chatbot class')

    def generate_response(self, processed_input):
        raise NotImplementedError('generate_response() not implemented in base Chatbot class')

    def respond(self, out_message = None):
        if isinstance(out_message, str): 
            print(out_message)

        received_input = self.receive_input()
        processed_input = self.process_input(received_input)
        response = self.generate_response(processed_input)
        return response