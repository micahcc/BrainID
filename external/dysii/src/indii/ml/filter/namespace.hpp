namespace indii {
  namespace ml {
    namespace filter {

/**
 * @namespace indii::ml::filter Filters and smoothers.
 *
 * @section estimator_usage Usage
 *
 * All the filters and smoothers provided have the same usage
 * idiom. This requires two objects:
 *
 * @li A @em %method object, which implements the particular
 * filtering or smoothing algorithm and stores the predicted system
 * state \f$\mathbf{x}\f$.

 * @li A @em model object, which defines the transition and
 * measurement functions \f$f\f$ and \f$g\f$ and system and
 * measurement noise \f$\mathbf{w}\f$ and \f$\mathbf{v}\f$.
 *
 * Start by choosing the method to use (e.g. a particle %filter), and
 * reading the documentation for the associated class
 * (e.g. ParticleFilter). Each method has a corresponding virtual
 * model class (e.g. ParticleFilterModel). Implement your model by
 * writing a new class that derives from this virtual class. The
 * virtual class provides the functions that must be implemented in
 * order that the model be compatible with your chosen method.
 *
 * Now, instantiate an object of your model class:
 *
 * @code
 * MyParticleFilterModel model;
 * @endcode
 *
 * and a prior distribution over the state of the system:
 *
 * @code
 * indii::ml::aux::vector mu(5);
 * indii::ml::aux::symmetric_matrix sigma(5);
 *
 * mu.clear();
 * mu(0) = -1.0;
 * mu(1) = 1.0;
 * mu(2) = 0.8;
 * mu(3) = 0.1;
 * mu(4) = 5e-3;
 *
 * sigma.clear();
 * sigma(0,0) = 1.0;
 * sigma(1,1) = 1.0;
 * sigma(2,2) = 0.01;
 * sigma(3,3) = 1e-6;
 * sigma(4,4) = 1e-6;
 *
 * indii::ml::aux::GaussianPdf tmp(mu, sigma);
 * indii::ml::aux::DiracMixturePdf x0(tmp, 500);
 * @endcode
 *
 * Pass both of these to the constructor of your chosen method to
 * instantiate it:
 *
 * @code
 * ParticleFilter filter(&model, x0);
 * @endcode
 *
 * Apply the method by using its step functions to advance it forwards
 * or backwards through time, passing in the relevant measurement at
 * each step. After each step, retrieve the estimated system state.
 *
 * @code
 * unsigned int T = 100;
 * unsigned int t;
 * double y[T] = { ... }; // measurements, perhaps read in from a file
 *
 * for (t = 1; t <= T; t++) {
 *   filter.filter(t, y[t]);
 *   std::cout << t << '=';
 *   std::cout << filter.getFilteredState().getExpectation() << std::endl;
 * }
 * @endcode
 * 
 * Smoothers have a similar usage idiom. First filter the data, then
 * initialise a smoother, such as ParticleSmoother, with the last
 * state of the filter, and use the smoother's stepping function to
 * proceed backward through time.
 *
 * See the test suite for more elaborate example code.
 *
 * @par Tip:
 * For the KalmanFilter, KalmanSmoother and RauchTungStriebelSmoother
 * methods, you may instantiate an object of the LinearModel class as
 * your model, rather than deriving your own model class.
 */

    }
  }
}
