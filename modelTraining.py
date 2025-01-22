
def train_model(model, train_data, test_data):
    history = model.fit(train_data, # use same training data created above
                            epochs=5,
                            steps_per_epoch=len(train_data),
                            validation_data=test_data, # use same validation data created above
                            validation_steps=len(test_data))
    return history