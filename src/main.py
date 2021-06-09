from data.make_dataset import get_data
from models.predict_model import evaluate, get_predictions
from models.train_model import train
from visualization import visualize

if __name__ == '__main__':
    # Train model
    print('Train model')
    train(model_name='test_model.pth', epochs=1)

    # Evaluate model
    print('Evaluate model')
    evaluate('../models/model.pth')

    # Predict
    print('Predict model')
    _, test_set = get_data('../data')

    images, labels = next(iter(test_set))

    pred = get_predictions('../models/model.pth', images)
    print('Accuracy =',
          (pred == labels.view(*pred.shape)).float().mean().item()
          )

    # Visualize
    visualize.tSNE('../models/model.pth')
