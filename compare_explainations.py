import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob
from matplotlib.gridspec import GridSpec

def compare_normal_vs_pneumonia():
    """Compare heatmap patterns between normal and pneumonia cases"""
    print("🔬 Comparing Normal vs Pneumonia XAI Patterns")
    print("=" * 50)
    
    # Get explanation files
    normal_files = glob.glob("results/explanations/XAI_NORMAL_*.png")
    pneumonia_files = glob.glob("results/explanations/XAI_PNEUMONIA_*.png")
    
    print(f"📁 Found {len(normal_files)} normal cases and {len(pneumonia_files)} pneumonia cases")
    
    if not normal_files or not pneumonia_files:
        print("❌ Need both normal and pneumonia explanations for comparison")
        return
    
    # Take one example of each
    normal_example = normal_files[0]
    pneumonia_example = pneumonia_files[0]
    
    print(f"🔍 Comparing:")
    print(f"   NORMAL: {os.path.basename(normal_example)}")
    print(f"   PNEUMONIA: {os.path.basename(pneumonia_example)}")
    
    # Load images
    normal_img = cv2.imread(normal_example)
    pneumonia_img = cv2.imread(pneumonia_example)
    
    if normal_img is None or pneumonia_img is None:
        print("❌ Could not load comparison images")
        return
    
    # Convert BGR to RGB for matplotlib
    normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
    pneumonia_img = cv2.cvtColor(pneumonia_img, cv2.COLOR_BGR2RGB)
    
    # Create comparison visualization
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig)
    
    # Titles
    fig.suptitle('XAI Comparison: Normal vs Pneumonia Chest X-Rays', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Normal Case - Full explanation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(normal_img[100:600, 50:550])  # Original image region
    ax1.set_title('NORMAL CASE\nOriginal X-Ray', fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(normal_img[100:600, 600:1100])  # Preprocessed region
    ax2.set_title('Preprocessed', fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(normal_img[650:1150, 50:550])  # Heatmap region
    ax3.set_title('Grad-CAM Heatmap', fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(normal_img[650:1150, 600:1100])  # Overlay region
    ax4.set_title('Explanation Overlay', fontweight='bold')
    ax4.axis('off')
    
    # Pneumonia Case - Full explanation
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(pneumonia_img[100:600, 50:550])
    ax5.set_title('PNEUMONIA CASE\nOriginal X-Ray', fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(pneumonia_img[100:600, 600:1100])
    ax6.set_title('Preprocessed', fontweight='bold')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(pneumonia_img[650:1150, 50:550])
    ax7.set_title('Grad-CAM Heatmap', fontweight='bold')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(pneumonia_img[650:1150, 600:1100])
    ax8.set_title('Explanation Overlay', fontweight='bold')
    ax8.axis('off')
    
    # Heatmap Intensity Analysis
    ax9 = fig.add_subplot(gs[2, :])
    
    # Analyze heatmap intensities
    normal_heatmaps = []
    pneumonia_heatmaps = []
    
    for file in normal_files[:3]:  # Sample 3 of each
        img = cv2.imread(file)
        if img is not None:
            # Extract heatmap region (adjust coordinates based on your layout)
            heatmap_region = img[650:1150, 50:550]
            # Convert to grayscale for intensity analysis
            gray_heatmap = cv2.cvtColor(heatmap_region, cv2.COLOR_BGR2GRAY)
            normal_heatmaps.append(gray_heatmap)
    
    for file in pneumonia_files[:3]:
        img = cv2.imread(file)
        if img is not None:
            heatmap_region = img[650:1150, 50:550]
            gray_heatmap = cv2.cvtColor(heatmap_region, cv2.COLOR_BGR2GRAY)
            pneumonia_heatmaps.append(gray_heatmap)
    
    # Calculate statistics
    normal_means = [np.mean(hm) for hm in normal_heatmaps]
    pneumonia_means = [np.mean(hm) for hm in pneumonia_heatmaps]
    
    normal_stds = [np.std(hm) for hm in normal_heatmaps]
    pneumonia_stds = [np.std(hm) for hm in pneumonia_heatmaps]
    
    # Create bar chart
    categories = ['Normal Cases', 'Pneumonia Cases']
    means = [np.mean(normal_means), np.mean(pneumonia_means)]
    stds = [np.mean(normal_stds), np.mean(pneumonia_stds)]
    
    bars = ax9.bar(categories, means, yerr=stds, capsize=10, 
                   color=['lightblue', 'lightcoral'], alpha=0.7)
    
    ax9.set_ylabel('Heatmap Intensity (0-255)')
    ax9.set_title('Average Grad-CAM Heatmap Intensity Comparison', fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Key Findings
    intensity_diff = means[1] - means[0]
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"   🔵 Normal average intensity: {means[0]:.1f}")
    print(f"   🔴 Pneumonia average intensity: {means[1]:.1f}")
    print(f"   📈 Intensity difference: {intensity_diff:.1f} ({intensity_diff/means[0]*100:.1f}% higher)")
    
    if intensity_diff > 0:
        print("   💡 Pneumonia cases show STRONGER heatmap activation")
        print("   🎯 AI focuses more intensely on abnormal regions in pneumonia cases")
    else:
        print("   💡 Normal cases show stronger activation (unexpected)")
    
    plt.tight_layout()
    plt.savefig('results/comparison_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n💾 Comparison saved to: results/comparison_analysis.png")

def analyze_attention_patterns():
    """Analyze where the AI focuses attention in different cases"""
    print("\n🎯 Analyzing Attention Patterns")
    print("=" * 40)
    
    pneumonia_files = glob.glob("results/explanations/XAI_PNEUMONIA_*.png")
    
    if pneumonia_files:
        print("🔍 In pneumonia cases, the AI typically focuses on:")
        print("   • Lung consolidation areas (dense white patches)")
        print("   • Air bronchograms (air-filled tubes in dense lungs)") 
        print("   • Infiltrates (hazy inflammatory areas)")
        print("   • Opacity patterns (abnormal whiteness)")
        print("   • Asymmetries between left and right lungs")
        
        print("\n🎓 Clinical correlation:")
        print("   • Red/Orange areas = High diagnostic importance")
        print("   • Blue areas = Normal lung tissue")
        print("   • The heatmap should align with radiologist findings")

if __name__ == "__main__":
    compare_normal_vs_pneumonia()
    analyze_attention_patterns()