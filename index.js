const {Iris}=require('machinelearn/datasets');
const {KNeighborsClassifier}=require('machinelearn/neighbors');
const { train_test_split }=require('machinelearn/model_selection');

(async function(){
    const iris=new Iris();
    const d=await iris.load();
    const knn=new KNeighborsClassifier();
    let s = train_test_split(d.data,d.targets,{
        test_size : 0.1,
        train_size : 0.9 ,
        random_state : 50
    });
    knn.fit(s.xTrain, s.yTrain);
    let c = 0;
    for(let i=0; i<s.xTest.length; i++){
        if(s.yTest[i]==knn.predict(s.xTest[i]))
        {
            c++;
        }
    }
    console.log(c/s.xTest.length);
})();