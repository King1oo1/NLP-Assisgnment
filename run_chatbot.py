from plot_twist_chatbot import PlotTwistChatbot

if __name__ == "__main__":
    # Create chatbot instance
    chatbot = PlotTwistChatbot()
    
    # Start the conversation
    chatbot.greeting()
    
    # Initial story generation
    initial_response = chatbot.generate_response({
        "action_type": "narrative", 
        "sentiment": "neutral",
        "keywords": [],
        "voice_emotion": "neutral"
    })
    print(f"\n{initial_response}")
    
    # Main conversation loop
    while chatbot.conversation_is_active:
        try:
            user_input = chatbot.receive_input()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                chatbot.conversation_is_active = False
                break
                
            # Process and respond to user input
            processed = chatbot.process_input(user_input)
            response = chatbot.generate_response(processed)
            print(f"\n{response}")
            
        except KeyboardInterrupt:
            print("\n\nStory interrupted...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Let's continue the story...")
            response = "What would you like to do next?"
            print(f"\n{response}")
    
    chatbot.farewell()