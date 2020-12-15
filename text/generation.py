"""
An algorithm to generate text is as follows:
1. Specify a seed string (e.g. 'ROMEO:') to get the network started, and a define number of characters for the model to generate, num_generation_steps.
2. Tokenize this sentence to obtain a list containing one list of the integer tokens.
3. Reset the initial state of the network.
4. Convert the token list into a Tensor (or numpy array) and pass it to your model as a batch of size one.
5. Get the model prediction (logits) for the last time step and extract the state of the recurrent layer.
6. Use the logits to construct a categorical distribution and sample a token from it.
7. Repeat the following for num_generation_steps - 1 steps:
    A. Use the saved state of the recurrent layer and the last sampled token to get new logit predictions
    B. Use the logits to construct a new categorical distribution and sample a token from it.
    C. Save the updated state of the recurrent layer.
8. Take the final list of tokens and convert to text using the Tokenizer.
"""

def get_logits(model, token_sequence, initial_state=None):
    """
    This function takes a model object, a token sequence and an optional initial
    state for the recurrent layer. The function should return the logits prediction
    for the final time step as a 2D numpy array.
    """
    input_tensor = tf.convert_to_tensor(token_sequence)
    model.layers[1].reset_states(initial_state)
    
    return model.predict(input_tensor)[:,-1,:]    

def sample_token(logits):
    """
    This function takes a 2D numpy array as an input, and constructs a 
    categorical distribution using it. It should then sample from this
    distribution and return the sample as a single integer.
    """
    return tf.random.categorical(logits, num_samples = 1).numpy()[0][0]

def get_model(vocab_size, batch_size):
    """
    This function takes a vocabulary size and batch size, and builds and returns a 
    Sequential model according to the above specification.
    """
#     input_layer = Input(shape = (None,), batch_size=batch_size)
#     x = Embedding(vocab_size,256,mask_zero=True)(input_layer)
#     x = GRU(units=1024, stateful=True, return_sequences=True)(x)
#     output = Dense(units=vocab_size)(x)
#     return Model(inputs = input_layer, outputs = output)

    return Sequential([
        Embedding(vocab_size,256,mask_zero=True,batch_input_shape = (batch_size, None)),
        GRU(units=1024, stateful=True, return_sequences=True),
        Dense(units=vocab_size)
    ])
    

def generate_strings(tokenizer,model,init_string = 'ROMEO:', num_generation_steps = 1000):
    

    token_sequence = tokenizer.texts_to_sequences([init_string])
    initial_state = None
    input_sequence = token_sequence

    for _ in range(num_generation_steps):
        logits = get_logits(model, input_sequence, initial_state=initial_state)
        sampled_token = sample_token(logits)
        token_sequence[0].append(sampled_token)
        input_sequence = [[sampled_token]]
        ## layers[1] is the GRU layer
        initial_state = model.layers[1].states[0].numpy()
        
    return tokenizer.sequences_to_texts(token_sequence)[0][::2]


