// 6/26/2025 AI-Tag
// This was created with the help of Assistant, a Unity Artificial Intelligence product.

// Note: The following errors are because the TextMeshPro package is not installed.
// Please install it from the Unity Package Manager.

using UnityEngine;
using UnityEngine.UI;


/// <summary>
    /// Manages the UI for the picture-in-picture replay system
    /// Handles showing/hiding the replay window and playback controls
    /// </summary>
    public class ReplayUI : MonoBehaviour
    {
        [Header("UI References")]
        [SerializeField] private Canvas replayCanvas;
        [SerializeField] private RawImage replayDisplay;
        [SerializeField] private CanvasGroup replayWindow;
        
        [Header("Control Elements")]
        [SerializeField] private Button playPauseButton;
        [SerializeField] private Button stopButton;
        [SerializeField] private Button closeButton;
        [SerializeField] private Slider timelineSlider;
        [SerializeField] private Slider speedSlider;
        
        [Header("Display Elements")]
        [SerializeField] private TextMeshProUGUI timeDisplay;
        [SerializeField] private TextMeshProUGUI speedDisplay;
        [SerializeField] private TextMeshProUGUI titleText;
        [SerializeField] private Image playPauseIcon;
        
        [Header("Icons")]
        [SerializeField] private Sprite playIcon;
        [SerializeField] private Sprite pauseIcon;
        
        [Header("Animation Settings")]
        [SerializeField] private float fadeInDuration = 0.3f;
        [SerializeField] private float fadeOutDuration = 0.2f;
        [SerializeField] private AnimationCurve fadeCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
        
        private ReplayManager replayManager;
        private ReplayPlayer replayPlayer;
        private bool isUIVisible = false;
        private Coroutine fadeCoroutine;

        private void Awake()
        {
            InitializeReferences();
            SetupEventListeners();
            HideReplayUI(immediate: true);
        }

        private void InitializeReferences()
        {
            // Get references to other replay components
            replayManager = GetComponent<ReplayManager>();
            replayPlayer = GetComponent<ReplayPlayer>();
            
            // Create default UI elements if not assigned
            if (replayCanvas == null)
            {
                replayCanvas = FindObjectOfType<Canvas>();
            }
            
            if (replayWindow == null && replayCanvas != null)
            {
                CreateDefaultReplayWindow();
            }
            
            // Validate UI references
            ValidateUIReferences();
        }

        private void CreateDefaultReplayWindow()
        {
            // TODO: Create a default replay UI window programmatically
            // This is a fallback if no UI is assigned in the inspector
            
            var windowGO = new GameObject("ReplayWindow");
            windowGO.transform.SetParent(replayCanvas.transform, false);
            
            replayWindow = windowGO.AddComponent<CanvasGroup>();
            
            var rectTransform = windowGO.GetComponent<RectTransform>();
            rectTransform.anchorMin = new Vector2(0.65f, 0.65f);
            rectTransform.anchorMax = new Vector2(0.95f, 0.95f);
            rectTransform.offsetMin = Vector2.zero;
            rectTransform.offsetMax = Vector2.zero;
            
            // Add background
            var background = windowGO.AddComponent<Image>();
            background.color = new Color(0, 0, 0, 0.8f);
            
            Debug.Log("[ReplayUI] Created default replay window");
        }

        private void ValidateUIReferences()
        {
            if (replayCanvas == null)
            {
                Debug.LogWarning("[ReplayUI] No Canvas assigned - UI will not function properly");
            }
            
            if (replayWindow == null)
            {
                Debug.LogWarning("[ReplayUI] No replay window assigned - creating placeholder");
            }
        }

        private void SetupEventListeners()
        {
            // Setup button listeners
            if (playPauseButton != null)
            {
                playPauseButton.onClick.AddListener(TogglePlayPause);
            }
            
            if (stopButton != null)
            {
                stopButton.onClick.AddListener(StopReplay);
            }
            
            if (closeButton != null)
            {
                closeButton.onClick.AddListener(CloseReplay);
            }
            
            if (timelineSlider != null)
            {
                timelineSlider.onValueChanged.AddListener(OnTimelineChanged);
            }
            
            if (speedSlider != null)
            {
                speedSlider.onValueChanged.AddListener(OnSpeedChanged);
                speedSlider.value = 1f; // Default speed
            }
            
            // Subscribe to replay events
            if (replayPlayer != null)
            {
                replayPlayer.OnPlaybackProgress += OnPlaybackProgress;
                replayPlayer.OnPlaybackComplete += OnPlaybackComplete;
            }
        }

        public void ShowReplayUI()
        {
            if (isUIVisible)
                return;

            isUIVisible = true;
            
            if (replayWindow != null)
            {
                replayWindow.gameObject.SetActive(true);
                
                if (fadeCoroutine != null)
                {
                    StopCoroutine(fadeCoroutine);
                }
                
                fadeCoroutine = StartCoroutine(FadeIn());
            }
            
            // Update UI elements
            UpdatePlayPauseButton();
            UpdateSpeedDisplay();
            UpdateTitleText();
            
            Debug.Log("[ReplayUI] Replay UI shown");
        }

        public void HideReplayUI(bool immediate = false)
        {
            if (!isUIVisible)
                return;

            isUIVisible = false;
            
            if (replayWindow != null)
            {
                if (immediate)
                {
                    replayWindow.alpha = 0f;
                    replayWindow.gameObject.SetActive(false);
                }
                else
                {
                    if (fadeCoroutine != null)
                    {
                        StopCoroutine(fadeCoroutine);
                    }
                    
                    fadeCoroutine = StartCoroutine(FadeOut());
                }
            }
            
            Debug.Log("[ReplayUI] Replay UI hidden");
        }

        private System.Collections.IEnumerator FadeIn()
        {
            float elapsedTime = 0f;
            
            while (elapsedTime < fadeInDuration)
            {
                float progress = elapsedTime / fadeInDuration;
                float alpha = fadeCurve.Evaluate(progress);
                
                if (replayWindow != null)
                {
                    replayWindow.alpha = alpha;
                }
                
                elapsedTime += Time.deltaTime;
                yield return null;
            }
            
            if (replayWindow != null)
            {
                replayWindow.alpha = 1f;
            }
        }

        private System.Collections.IEnumerator FadeOut()
        {
            float elapsedTime = 0f;
            float startAlpha = replayWindow != null ? replayWindow.alpha : 1f;
            
            while (elapsedTime < fadeOutDuration)
            {
                float progress = elapsedTime / fadeOutDuration;
                float alpha = Mathf.Lerp(startAlpha, 0f, fadeCurve.Evaluate(progress));
                
                if (replayWindow != null)
                {
                    replayWindow.alpha = alpha;
                }
                
                elapsedTime += Time.deltaTime;
                yield return null;
            }
            
            if (replayWindow != null)
            {
                replayWindow.alpha = 0f;
                replayWindow.gameObject.SetActive(false);
            }
        }

        private void TogglePlayPause()
        {
            if (replayPlayer == null)
                return;

            if (replayPlayer.IsPlaying)
            {
                replayPlayer.PausePlayback();
            }
            else
            {
                replayPlayer.ResumePlayback();
            }
            
            UpdatePlayPauseButton();
        }

        private void StopReplay()
        {
            if (replayPlayer != null)
            {
                replayPlayer.StopPlayback();
            }
        }

        private void CloseReplay()
        {
            if (replayManager != null)
            {
                replayManager.StopReplay();
            }
            else
            {
                HideReplayUI();
            }
        }

        private void OnTimelineChanged(float value)
        {
            if (replayPlayer != null)
            {
                replayPlayer.SeekToTime(value);
            }
        }

        private void OnSpeedChanged(float value)
        {
            if (replayPlayer != null)
            {
                replayPlayer.PlaybackSpeed = value;
            }
            
            UpdateSpeedDisplay();
        }

        private void OnPlaybackProgress(float progress)
        {
            if (timelineSlider != null)
            {
                timelineSlider.SetValueWithoutNotify(progress);
            }
            
            UpdateTimeDisplay();
        }

        private void OnPlaybackComplete()
        {
            UpdatePlayPauseButton();
            
            // Auto-close after a short delay
            Invoke(nameof(CloseReplay), 1f);
        }

        private void UpdatePlayPauseButton()
        {
            if (playPauseButton == null || playPauseIcon == null)
                return;

            bool isPlaying = replayPlayer != null && replayPlayer.IsPlaying;
            
            if (isPlaying && pauseIcon != null)
            {
                playPauseIcon.sprite = pauseIcon;
            }
            else if (!isPlaying && playIcon != null)
            {
                playPauseIcon.sprite = playIcon;
            }
        }

        private void UpdateTimeDisplay()
        {
            if (timeDisplay == null || replayPlayer == null)
                return;

            float currentTime = replayPlayer.GetCurrentTime();
            float totalTime = replayPlayer.GetTotalDuration();
            
            string timeText = $"{FormatTime(currentTime)} / {FormatTime(totalTime)}";
            timeDisplay.text = timeText;
        }

        private void UpdateSpeedDisplay()
        {
            if (speedDisplay == null || replayPlayer == null)
                return;

            float speed = replayPlayer.PlaybackSpeed;
            speedDisplay.text = $"{speed:F1}x";
        }

        private void UpdateTitleText()
        {
            if (titleText != null)
            {
                titleText.text = "Instant Replay";
            }
        }

        private string FormatTime(float timeInSeconds)
        {
            int minutes = Mathf.FloorToInt(timeInSeconds / 60f);
            int seconds = Mathf.FloorToInt(timeInSeconds % 60f);
            return $"{minutes:00}:{seconds:00}";
        }

        private void Update()
        {
            if (isUIVisible)
            {
                UpdateTimeDisplay();
            }
        }

        private void OnDestroy()
        {
            // Clean up event listeners
            if (playPauseButton != null)
            {
                playPauseButton.onClick.RemoveListener(TogglePlayPause);
            }
            
            if (stopButton != null)
            {
                stopButton.onClick.RemoveListener(StopReplay);
            }
            
            if (closeButton != null)
            {
                closeButton.onClick.RemoveListener(CloseReplay);
            }
            
            if (timelineSlider != null)
            {
                timelineSlider.onValueChanged.RemoveListener(OnTimelineChanged);
            }
            
            if (speedSlider != null)
            {
                speedSlider.onValueChanged.RemoveListener(OnSpeedChanged);
            }
            
            if (replayPlayer != null)
            {
                replayPlayer.OnPlaybackProgress -= OnPlaybackProgress;
                replayPlayer.OnPlaybackComplete -= OnPlaybackComplete;
            }
        }

        // Public properties
        public bool IsUIVisible => isUIVisible;
        public Canvas ReplayCanvas => replayCanvas;
        public RawImage ReplayDisplay => replayDisplay;
    }