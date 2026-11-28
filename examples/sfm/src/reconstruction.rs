//! Structure-from-Motion reconstruction wrapper.
//!
//! Builds a [`PinholeCamera`] per video frame from user-supplied intrinsics
//! and drives `kornia_calib`'s incremental SfM (`reconstruct`) over the
//! feature tracks. Because no AprilTag anchors the scale, the recovered map is
//! [`ScaleSource::UpToScale`]: correct shape, arbitrary units.

use std::error::Error;
use std::sync::Arc;

use kornia_3d::camera::PinholeCamera;
use kornia_calib::{reconstruct, FeatureTrack, Reconstruction, ReconstructionConfig};

/// Build a pinhole camera from intrinsics, with zero distortion.
pub(crate) fn make_camera(fx: f64, fy: f64, cx: f64, cy: f64) -> PinholeCamera {
    PinholeCamera {
        fx,
        fy,
        cx,
        cy,
        ..PinholeCamera::IDENTITY
    }
}

/// Run incremental SfM over the given feature tracks.
///
/// Each video frame is treated as one camera with the given intrinsics; the
/// `tracks[i].obs` camera indices index into that per-frame camera list. No
/// tag is supplied, so the reconstruction is up to scale.
///
/// # Arguments
///
/// * `tracks` - Multi-view feature tracks produced by `kornia_calib::build_tracks`.
/// * `fx` / `fy` - Focal lengths in pixels.
/// * `cx` / `cy` - Principal point in pixels.
/// * `n_frames` - Number of video frames (one camera per frame).
/// * `progress` - Optional callback fired after each registration, as
///   `(registered_so_far, total_views)`.
/// * `config` - Reconstruction tuning; start from
///   `ReconstructionConfig::new(0.0).sequential()` for video walkthroughs.
///
/// # Returns
///
/// The [`Reconstruction`] with per-frame poses and reconstructed 3D points.
///
/// # Errors
///
/// Returns an error if the reconstruction cannot bootstrap (e.g. no usable
/// tracks, or a degenerate camera configuration).
#[allow(clippy::too_many_arguments)] // thin wrapper over reconstruct's (also 8-arg) signature
pub fn run_sfm(
    tracks: &[FeatureTrack],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    n_frames: usize,
    progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
    mut config: ReconstructionConfig,
) -> Result<Reconstruction, Box<dyn Error>> {
    let cameras = vec![make_camera(fx, fy, cx, cy); n_frames];
    config.progress = progress;
    Ok(reconstruct(&cameras, &[], tracks, &config, None)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;
    use kornia_calib::ScaleSource;

    #[test]
    fn make_camera_constructs_pinhole_camera() {
        let cam = make_camera(500.0, 520.0, 320.0, 240.0);
        assert_eq!(cam.fx, 500.0);
        assert_eq!(cam.fy, 520.0);
        assert_eq!(cam.cx, 320.0);
        assert_eq!(cam.cy, 240.0);
        assert_eq!(cam.k1, 0.0);
        assert_eq!(cam.k2, 0.0);
        assert_eq!(cam.p1, 0.0);
        assert_eq!(cam.p2, 0.0);
    }

    #[test]
    fn run_sfm_recovers_points_and_poses_from_synthetic_scene() {
        use kornia_algebra::linalg::rigid::umeyama_f64;
        use kornia_algebra::Vec3F64;

        let tracks = test_util::synthetic_tracks();

        let recon = run_sfm(
            &tracks,
            test_util::FX,
            test_util::FY,
            test_util::CX,
            test_util::CY,
            test_util::N_FRAMES,
            None,
            ReconstructionConfig::new(0.0).sequential(),
        )
        .expect("synthetic scene must reconstruct");

        assert!(
            !recon.points.is_empty(),
            "reconstruction should recover 3D points"
        );

        // The bootstrap pair alone satisfies ">= 2"; require all three cameras,
        // which is what exercising PnP registration actually means.
        let recovered: Vec<Vec3F64> = recon
            .views
            .iter()
            .map(|v| v.as_ref().map(|p| p.translation))
            .collect::<Option<Vec<_>>>()
            .expect("all three cameras must register");

        // The map is up to scale and world frame; align the recovered camera
        // centres to ground truth with a similarity before comparing.
        let gt: Vec<Vec3F64> = test_util::synthetic_poses()
            .iter()
            .map(|p| p.inverse().translation)
            .collect();
        let alignment = umeyama_f64(&recovered, &gt, None, true).expect("similarity alignment");
        let (r, t, s) = (alignment.r, alignment.t, alignment.s);
        let max_err = recovered
            .iter()
            .zip(&gt)
            .map(|(p, q)| (r * *p * s + t - *q).length())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-6,
            "similarity-aligned camera centres must match ground truth (max err {max_err:.3e})"
        );
    }

    #[test]
    fn run_sfm_is_up_to_scale_without_tag() {
        let tracks = test_util::synthetic_tracks();

        let recon = run_sfm(
            &tracks,
            test_util::FX,
            test_util::FY,
            test_util::CX,
            test_util::CY,
            test_util::N_FRAMES,
            None,
            ReconstructionConfig::new(0.0).sequential(),
        )
        .expect("synthetic scene must reconstruct");
        assert_eq!(recon.scale, ScaleSource::UpToScale);
    }
}
