namespace SDRClassifier
{
    using NumSharp;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    public class SDRClassifier
    {

        public int version = 1;
        public List<int> steps;
        public float alpha;
        public float actValueAlpha;
        public float verbosity;
        private int _maxSteps;
        private List<int> _patternNZHistory = new();
        private int _maxInputIdx;
        private int _maxBucketIdx;
        private Dictionary<object, NDArray> _weightMatrix = new();
        private List<object> _actualValues = new();

        public SDRClassifier(List<int> steps, float alpha, float actValueAlpha, float verbosity, int version)
        {
            if (steps.Count == 0)
            {
                Console.WriteLine("steps cannot be empty");
            }
            if (alpha < 0)
            {
                Console.WriteLine("alpha (learning rate) must be a positive number");
            }
            if (actValueAlpha < 0 || actValueAlpha >= 1)
            {
                Console.WriteLine("actValueAlpha be a number between 0 and 1");
            }

            // Save constructor args
            this.steps = steps;
            this.alpha = alpha;
            this.actValueAlpha = actValueAlpha;
            this.verbosity = verbosity;
            // Max # of steps of prediction we need to support
            this._maxSteps = this.steps.Max() + 1;
            // History of the last _maxSteps activation patterns. We need to keep
            // these so that we can associate the current iteration's classification
            // with the activationPattern from N steps ago
            //this._patternNZHistory = this.deque(maxlen: this._maxSteps);
            // This contains the value of the highest input number we've ever seen
            // It is used to pre-allocate fixed size arrays that hold the weights
            this._maxInputIdx = 0;
            // This contains the value of the highest bucket index we've ever seen
            // It is used to pre-allocate fixed size arrays that hold the weights of
            // each bucket index during inference
            this._maxBucketIdx = 0;
            // The connection weight matrix
            this._weightMatrix = new Dictionary<object, NDArray>();
            foreach (var step in this.steps)
            {
                this._weightMatrix[step] = np.zeros(shape: (this._maxInputIdx + 1, this._maxBucketIdx + 1));
            }
            // This keeps track of the actual value to use for each bucket index. We
            // start with 1 bucket, no actual value so that the first infer has something
            // to return
            this._actualValues = new List<object> { null };
            // Set the version to the latest version.
            // This is used for serialization/deserialization
            this.version = version;
        }

        private List<int> deque(int maxlen)
        {
            throw new NotImplementedException();
        }

        public void Compute(
                int recordNum,
                List<int> patternNZ,
                Dictionary<string, object> classification,
                bool learn,
                object infer)
        {
            //int nSteps;
            object numCategory;
            object actValueList;
            object bucketIdxList;
            if (this.verbosity >= 1)
            {
                Console.WriteLine("  learn:", learn);
                Console.WriteLine("  recordNum:", recordNum);
                //Console.WriteLine(String.Format("  patternNZ (%d):", patternNZ.Count), patternNZ);
                Console.WriteLine("  classificationIn:", classification);
            }
            // ensures that recordNum increases monotonically
            if (this._patternNZHistory.Count > 0)
            {
                /*
                if (recordNum < this._patternNZHistory[-1][0])
                {
                    throw ValueError("the record number has to increase monotonically");
                }*/
            }
            // Store pattern in our history if this is a new record
            /*
            if (this._patternNZHistory.Count == 0 || recordNum > this._patternNZHistory[-1][0])
            {
                this._patternNZHistory.Add((recordNum, patternNZ));
            }
            */
            // To allow multi-class classification, we need to be able to run learning
            // without inference being on. So initialize retval outside
            // of the inference block.
            var retval = new Dictionary<object, object>
            {
            };
            // Update maxInputIdx and augment weight matrix with zero padding
            if (patternNZ.Max() > this._maxInputIdx)
            {
                var newMaxInputIdx = patternNZ.Max();
                foreach (var nSteps in this.steps)
                {
                    this._weightMatrix[nSteps] = np.concatenate(((NDArray, NDArray))(this._weightMatrix[nSteps], np.zeros(shape: (newMaxInputIdx - this._maxInputIdx, this._maxBucketIdx + 1))), axis: 0);
                }
                this._maxInputIdx = Convert.ToInt32(newMaxInputIdx);
            }
            // Get classification info
            if (classification is not null)
            {
                if (classification["bucketIdx"].GetType() != typeof(List<>))
                {
                    bucketIdxList = new List<object> { classification["bucketIdx"] };
                    actValueList = new List<object> { classification["actValue"] };
                    numCategory = 1;
                }
                else
                {
                    bucketIdxList = classification["bucketIdx"];
                    actValueList = classification["actValue"];
                    numCategory = ((List<object>)classification["bucketIdx"]).Count();
                }
            }
            else
            {
                if (learn)
                {
                    //throw ValueError("classification cannot be None when learn=True");
                    Console.WriteLine("classification cannot be None when learn=True");

                }
                actValueList = null;
                bucketIdxList = null;
            }
        }

        /// <summary>
        /// Return the inference value from one input sample. The actual
        /// learning happens in compute().
        /// </summary>
        /// <param name="patternNZ">list of the active indices from the output below</param>
        /// <param name="actValueList">
        /// dict of the classification information: bucketIdx: index of the encoder bucket actValue: actual value going into the encoder
        /// </param>
        /// <returns>
        /// dict containing inference results, one entry for each step in
        /// self.steps. The key is the number of steps, the value is an
        /// array containing the relative likelihood for each bucketIdx
        /// starting from bucketIdx 0.
        /// </returns>
        public object Infer(List<int> patternNZ, List<object> actValueList)
        {
            object defaultValue;
            /**
             * Return value dict. For buckets which we don't have an actual value
             * for yet, just plug in any valid actual value. It doesn't matter what
             * we use because that bucket won't have non-zero likelihood anyways.
             * NOTE: If doing 0-step prediction, we shouldn't use any knowledge
             * of the classification input during inference.
            */
            if (this.steps[0] == 0 || actValueList == null)
            {
                defaultValue = 0;
            }
            else
            {
                defaultValue = actValueList[0];
            }
            var actValues = (from x in this._actualValues
                             select x != null ? x : defaultValue).ToList();
            var retval = new Dictionary<object, object> { { "actualValues", actValues } };
            foreach (var nSteps in this.steps)
            {
                var predictDist = this.InferSingleStep(patternNZ, this._weightMatrix[nSteps]);
                retval[nSteps] = predictDist;
            }
            return retval;
        }

        /// <summary>
        /// Perform inference for a single step. Given an SDR input and a weight matrix, return a predicted distribution.
        /// </summary>
        /// <param name="patternNZ">list of the active indices from the output below</param>
        /// <param name="weightMatrix">Multidimentional array of the weight matrix</param>
        /// <returns>
        /// Multidimentional array of the predicted class label distribution
        /// </returns>
        public object InferSingleStep(List<int> patternNZ, NDArray weightMatrix)
        {
            var outputActivation = weightMatrix[patternNZ].sum(axis: 0);
            // softmax normalization
            outputActivation = outputActivation - np.max(outputActivation);
            var expOutputActivation = np.exp(outputActivation);
            var predictDist = expOutputActivation / np.sum(expOutputActivation);
            return predictDist;
        }
    }
}