const {Boston}=require('machinelearn/datasets');
const {LinearRegression}=require('machinelearn/linear_model');
const { train_test_split }=require('machinelearn/model_selection');
const { mean_squared_error } = require('machinelearn/metrics');
const tf=require('@tensorflow/tfjs');
tf.disableDeprecationWarnings();

(async function(){
    const boston=new Boston();
    const d=await boston.load();
    const ln = new LinearRegression();
    //console.log(d.data);
    let s=train_test_split(d.data, d.targets, {
        test_size: 0.3,
        train_size: 0.7,
        random_state: Math.random()*100
    });
    ln.fit(s.xTrain, s.yTrain);
    let y= [];
    let ynew = [];
    for(let i=0; i<s.xTest.length; i++){
        y.push(s.yTest[i]);
        ynew.push(ln.predict([s.xTest[i]])[0]);
        console.log(y[i]+"--"+ynew[i]);
    }
    console.log(mean_squared_error(y,ynew));
})();