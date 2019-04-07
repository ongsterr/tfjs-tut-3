import React, { Component } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

import houseData from './data.js'

class App extends Component {
  async componentDidMount() {
    await this.run()
  }

  createModel() {
    const model = tf.sequential()

    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))
    model.add(tf.layers.dense({ units: 1, useBias: true }))

    return model
  }

  convertToTensor(data) {
    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(data)

      const inputs = data.map(d => d.noOfBedroom)
      const labels = data.map(d => d.housePrice)

      // Step 2: Convert data to tensor
      const inputTensor = tf.tensor1d(inputs)
      const labelTensor = tf.tensor1d(labels)

      return {
        inputs: inputTensor,
        labels: labelTensor,
      }
    })
  }

  async trainModel(model, inputs, labels) {
    //Prepare model for training
    model.compile({
      optimizer: 'sgd',
      loss: 'meanSquaredError',
      metrics: 'mse',
    })

    const batchSize = 20
    const epochs = 20

    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 300, callbacks: ['onEpochEnd'] }
      ),
    })
  }

  testModel(model, inputData) {
    const [xs, preds] = tf.tidy(() => {
      const xs = tf.linspace(0, 20, 20)
      const preds = model.predict(xs.reshape([20, 1]))

      return [xs.dataSync(), preds.dataSync()]
    })

    const predictedPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] }
    })
    console.log(predictedPoints)

    const originalPoints = inputData.map(d => ({
      x: d.noOfBedroom,
      y: d.housePrice,
    }))

    tfvis.render.scatterplot(
      { name: 'Model Predictions vs Original Data' },
      {
        values: [originalPoints, predictedPoints],
        series: ['original', 'predicted'],
      },
      {
        xLabel: 'No of Bedroom',
        yLabel: 'House Price',
        height: 300,
      }
    )
  }

  async run() {
    const values = houseData.map(d => ({
      x: d.noOfBedroom,
      y: d.housePrice,
    }))

    // Render original data
    tfvis.render.scatterplot(
      { name: 'No of Bedrooms vs House Price' },
      { values },
      {
        xLabel: 'No of Bedrooms',
        yLabel: 'House Price',
        height: 300,
      }
    )

    // Create the model
    const model = this.createModel()
    tfvis.show.modelSummary({ name: 'Model Summary' }, model)

    const tensorData = this.convertToTensor(houseData)
    const { inputs, labels } = tensorData

    // Train the model
    await this.trainModel(model, inputs, labels)
    console.log('Done training')

    // Make some predictions using the model and compare them to original data
    this.testModel(model, houseData)
  }

  render() {
    return <div className="App">Hello Tensorflow</div>
  }
}

export default App
