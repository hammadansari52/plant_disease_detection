var express = require('express');
var app = express();
var fs = require('fs');
var bodyParser = require('body-parser');
var multer = require('multer');
const tf = require('@tensorflow/tfjs');
const tfn = require('@tensorflow/tfjs-node');
const upload = multer({dest: __dirname + '/uploads/images'});

app.set("view engine", "ejs");
app.use(bodyParser.urlencoded({extended: true}));
app.use(express.static('public'));


//===============================================
//MACHINE LEARNING PART--------------------------
//===============================================

async function init(model_name) {
	const handler = tfn.io.fileSystem("./models/" + model_name + "/model.json");
	const model = await tfn.loadGraphModel(handler);
	console.log("model loaded");
	return model;
}


async function init1(imageBuffer, model_name) {
	const model = await init(model_name);
	var tfimage = tfn.node.decodeImage(imageBuffer);
	tfimage = tfimage.as4D(-1,256,256,3);
	tfimage = tfimage.asType('float32');
	tfimage_min = tfimage.min();
	tfimage_max = tfimage.max();
	const normalized_tfimage = tfimage.sub(tfimage_min).div(tfimage_max.sub(tfimage_min));
	var predictions = await model.predict(normalized_tfimage);
	console.log("PREDICTION:", predictions.dataSync());
	return predictions.dataSync();
}

//===============================================
//***********************************************
//===============================================

app.get('/',function(req,res) {
    res.render('home');
});

app.get('/plants',function(req,res) {
    res.render('plants');
});

app.get('/plants/:plant_name',function(req,res) {
    res.render('show', {plant_name: req.params.plant_name});
});

app.post('/plants/:plant_name', upload.single('image'), async function (req, res) {
  console.log(req.file.filename);
  file_name = req.file.filename;
  var img_path = "./uploads/images/"+file_name;
  imageBuffer = fs.readFileSync(img_path);
  var pred = await init1(imageBuffer, req.params.plant_name);
  console.log(pred);
  res.render('result', {predictions:pred, plant_name: req.params.plant_name});
});

app.listen(8000, function(){
	console.log("Server started.............");
});