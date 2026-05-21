use numpy::ndarray::{Array2, ArrayView2, Axis};

/// Zero-copy view into training data.
pub struct DataView<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView2<'a, f64>,
}

impl<'a> DataView<'a> {
    pub fn new(x: ArrayView2<'a, f64>, y: ArrayView2<'a, f64>) -> Self {
        debug_assert_eq!(
            x.nrows(),
            y.ncols(),
            "X rows ({}) must match Y columns ({}).",
            x.nrows(),
            y.ncols()
        );
        Self { x, y }
    }

    pub fn n_samples(&self) -> usize {
        self.x.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }

    pub fn n_outputs(&self) -> usize {
        self.y.nrows()
    }
}

/// Owned data for storing in long-lived structs (e.g. Python bindings).
#[derive(Clone, Debug)]
pub struct OwnedData {
    pub x: Array2<f64>,
    pub y: Array2<f64>,
}

impl OwnedData {
    pub fn new(x: Array2<f64>, y: Array2<f64>) -> Self {
        debug_assert_eq!(x.nrows(), y.ncols());
        Self { x, y }
    }

    pub fn view(&self) -> DataView<'_> {
        DataView::new(self.x.view(), self.y.view())
    }

    pub fn n_samples(&self) -> usize {
        self.x.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }

    /* 
    Input is (n_outputs, n_samples), returns the first output. There should be a cleaner way to do this. 
    This computes the std over samples axis when initializing the kernel. 
    Should not be used when actually updating the running std since then each leaf output dimension must have distinct std.
    Thus, a future TODO is to maybe assert that all std values are the same over the n_outputs axis.
    */
    pub fn y_std_as_scalar(&self) -> f64 {
        self.y.std_axis(Axis(1), 0.0)[0] 
    }
    
}
