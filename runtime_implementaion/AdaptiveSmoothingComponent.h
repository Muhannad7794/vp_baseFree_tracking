
Folder highlights
Code defines an UAdaptiveSmoothingComponent class using C++ and Unreal Engine structs for managing adaptive smoothing based on movement kinematics.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
// CRITICAL: This include MUST be the last one in the list!
#include "AdaptiveSmoothingComponent.generated.h"

// 1. Define the Model Types
UENUM(BlueprintType)
enum class EAdaptiveModel : uint8
{
    NoSmoothing UMETA(DisplayName = "No Smoothing (Raw)"),
    Linear      UMETA(DisplayName = "Linear"),
    Piecewise   UMETA(DisplayName = "Piecewise"),
    Sigmoid     UMETA(DisplayName = "Sigmoid")
};

// 2. Configuration Structs
USTRUCT(BlueprintType)
struct FLinearAxisConfig
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere)
    float MinSigma = 0.0f;
    UPROPERTY(EditAnywhere)
    float MaxSigma = 100.0f;
    UPROPERTY(EditAnywhere)
    float MinSpeed = 3.0f;
    UPROPERTY(EditAnywhere)
    float MaxSpeed = 20.0f;
};

USTRUCT(BlueprintType)
struct FSigmoidAxisConfig
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere)
    float MinSigma = 0.0f;
    UPROPERTY(EditAnywhere)
    float MaxSigma = 100.0f;
    UPROPERTY(EditAnywhere)
    float MinSpeed = 3.0f;
    UPROPERTY(EditAnywhere)
    float MaxSpeed = 20.0f;
    UPROPERTY(EditAnywhere)
    float Midpoint = 50.0f;
    UPROPERTY(EditAnywhere)
    float Steepness = 0.1f;
};

USTRUCT(BlueprintType)
struct FPiecewiseAxisConfig
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere)
    TArray<float> SigmaBreaks;
    UPROPERTY(EditAnywhere)
    TArray<float> SpeedLevels;
};

// 3. The Class Declaration
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class VP_BASEFREE_TRACKING_API UAdaptiveSmoothingComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UAdaptiveSmoothingComponent();

    // -- MAIN INTERFACE --
    UFUNCTION(BlueprintCallable, Category = "Adaptive Smoothing")
    void GetSmoothedTransform(float DeltaTime, FVector RawLocation, FRotator RawRotation, FVector &OutSmoothedLoc, FRotator &OutSmoothedRot);

    // Dynamic Switcher
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Smoothing Control")
    EAdaptiveModel CurrentModel = EAdaptiveModel::NoSmoothing;

    // -- CONFIGURATION --
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config | General")
    int32 WindowSize = 25;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config | Sensor")
    float TargetCameraFPS = 25.0f; // Default to 25, but changable in Editor

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config | Linear")
    TArray<FLinearAxisConfig> LinearConfig;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config | Sigmoid")
    TArray<FSigmoidAxisConfig> SigmoidConfig;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config | Piecewise")
    TArray<FPiecewiseAxisConfig> PiecewiseConfig;

    // Shader reference
    // 1. Assign the static Mesh in the Editor
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Visualization")
    UStaticMeshComponent *CalibrationObject;

    // Material Instance Dynamic reference (created at runtime)
    UPROPERTY()
    UMaterialInstanceDynamic *DynamicMaterial;

private:
    // -- INTERNAL STATE --
    TArray<TArray<float>> AccelerationBuffers;

    // Internal Memory
    FVector InternalSmoothedLoc;
    FRotator InternalSmoothedRot;
    bool bFirstFrame = true;

    // Kinematics History
    FVector PrevLocation;
    FVector PrevLinearVelocity;
    FRotator PrevRotation;
    FVector PrevAngularVelocity;

    // Helpers
    float CalculateSigma(int32 AxisIndex, float CurrentAccel);
    float GetSpeed_Linear(int32 AxisIndex, float Sigma);
    float GetSpeed_Sigmoid(int32 AxisIndex, float Sigma);
    float GetSpeed_Piecewise(int32 AxisIndex, float Sigma);

    void InitializeBuffers();
    // Intialize the shader material
    void SetupMaterial();

    bool bIsInitialized = false;
};