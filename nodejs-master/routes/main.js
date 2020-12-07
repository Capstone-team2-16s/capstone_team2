var express = require('express');
var router = express.Router();
const { Target } = require('../models/Target');

router.get('/',function(req,res){
    res.render('index');
});
router.get('/targetInfoData',(req,res)=>{
    Target.find().sort({name:1}).exec(function(err, targets){
        if(err) {
          return res.status(500).send({error: err.message});
        }
        res.status(200).json(targets);
       });
});


module.exports = router;