
Folder highlights
Code defines an UAdaptiveSmoothingComponent class using C++ and Unreal Engine structs for managing adaptive smoothing based on movement kinematics.

#include "AdaptiveSmoothingComponent.h"
#include "Kismet/KismetMathLibrary.h"

UAdaptiveSmoothingComponent::UAdaptiveSmoothingComponent()
{
    PrimaryComponentTick.bCanEverTick = false;

    LinearConfig.SetNum(6);
    SigmoidConfig.SetNum(6);
    PiecewiseConfig.SetNum(6);
}

void UAdaptiveSmoothingComponent::InitializeBuffers()
{
    AccelerationBuffers.Empty();
    for (int i = 0; i < 6; i++)
    {
        AccelerationBuffers.Add(TArray<float>());
        AccelerationBuffers[i].Reserve(WindowSize + 5);
    }
    bIsInitialized = true;
}

// Initialize the shader Material to response to camera stability (Sigma)
void UAdaptiveSmoothingComponent::SetupMaterial()
{
    if (CalibrationObject)
    {
        // Create a dynamic instance of the material so we can change parameters
        DynamicMaterial = CalibrationObject->CreateAndSetMaterialInstanceDynamic(0);
    }
}

void UAdaptiveSmoothingComponent::GetSmoothedTransform(float DeltaTime, FVector RawLocation, FRotator RawRotation, FVector &OutSmoothedLoc, FRotator &OutSmoothedRot)
{
    // 1. Initialize or Reset
    if (!bIsInitialized)
        InitializeBuffers();

    // First Frame Snap
    if (bFirstFrame)
    {
        InternalSmoothedLoc = RawLocation;
        InternalSmoothedRot = RawRotation;
        PrevLocation = RawLocation;
        PrevRotation = RawRotation;
        bFirstFrame = false;
    }

    if (DeltaTime <= 0.0001f)
    {
        OutSmoothedLoc = InternalSmoothedLoc;
        OutSmoothedRot = InternalSmoothedRot;
        return;
    }

    // Fixing nDisplay / Lag Stability
    // Target Frame Time for 25 FPS = 1/25 = 0.04s
    // If the game lags (DT > 0.04), we clamp it to 0.04 to prevent acceleration from vanishing.
    // If the game runs fast (DT < 0.04), we use the real (smaller) DT for accuracy.
    const float TargetDT = 1.0f / TargetCameraFPS;
    float ClampedDT = FMath::Min(DeltaTime, TargetDT);

    // 2. MODEL: NO SMOOTHING (Bypass)
    if (CurrentModel == EAdaptiveModel::NoSmoothing)
    {
        InternalSmoothedLoc = RawLocation;
        InternalSmoothedRot = RawRotation;

        // Reset Kinematics to avoid spikes when switching back
        PrevLocation = RawLocation;
        PrevRotation = RawRotation;
        PrevLinearVelocity = FVector::ZeroVector;
        PrevAngularVelocity = FVector::ZeroVector;

        OutSmoothedLoc = RawLocation;
        OutSmoothedRot = RawRotation;
        return;
    }

    // 3. Calculate Kinematics (USING ClampedDT)
    // We use ClampedDT here so that a lag spike doesn't result in tiny acceleration.
    FVector CurrVel = (RawLocation - PrevLocation) / ClampedDT;
    FVector CurrAccel = (CurrVel - PrevLinearVelocity) / ClampedDT;
    PrevLocation = RawLocation;
    PrevLinearVelocity = CurrVel;

    FRotator DeltaRot = (RawRotation - PrevRotation).GetNormalized();
    FVector CurrAngVel;
    CurrAngVel.X = DeltaRot.Roll / ClampedDT;
    CurrAngVel.Y = DeltaRot.Pitch / ClampedDT;
    CurrAngVel.Z = DeltaRot.Yaw / ClampedDT;
    FVector CurrAngAccel = (CurrAngVel - PrevAngularVelocity) / ClampedDT;
    PrevRotation = RawRotation;
    PrevAngularVelocity = CurrAngVel;

    // 4. Calculate Speeds
    float Accels[6] = {(float)CurrAccel.X, (float)CurrAccel.Y, (float)CurrAccel.Z, (float)CurrAngAccel.X, (float)CurrAngAccel.Y, (float)CurrAngAccel.Z};
    float Speeds[6];
    // accumulate average sigma for shader update
    float TotalSigma = 0.0f;

    for (int i = 0; i < 6; i++)
    {
        // Pass the accel calculated with ClampedDT to the sigma function
        float Sigma = CalculateSigma(i, Accels[i]);

        // Accumulate Sigma for the first 3 axes (Position) for visualization
        if (i < 3)
            TotalSigma += Sigma;

        // Calculate speed based on each model
        switch (CurrentModel)
        {
        case EAdaptiveModel::Linear:
            Speeds[i] = GetSpeed_Linear(i, Sigma);
            break;
        case EAdaptiveModel::Sigmoid:
            Speeds[i] = GetSpeed_Sigmoid(i, Sigma);
            break;
        case EAdaptiveModel::Piecewise:
            Speeds[i] = GetSpeed_Piecewise(i, Sigma);
            break;
        default:
            Speeds[i] = 20.0f;
        }
    }

    // 5. Apply Smoothing (USING DeltaTime)
    // Still using REAL DeltaTime for the actual interpolation (movement) step.
    InternalSmoothedLoc.X = FMath::FInterpTo(InternalSmoothedLoc.X, RawLocation.X, DeltaTime, Speeds[0]);
    InternalSmoothedLoc.Y = FMath::FInterpTo(InternalSmoothedLoc.Y, RawLocation.Y, DeltaTime, Speeds[1]);
    InternalSmoothedLoc.Z = FMath::FInterpTo(InternalSmoothedLoc.Z, RawLocation.Z, DeltaTime, Speeds[2]);

    float NewRoll = FMath::FInterpTo(InternalSmoothedRot.Roll, RawRotation.Roll, DeltaTime, Speeds[3]);
    float NewPitch = FMath::FInterpTo(InternalSmoothedRot.Pitch, RawRotation.Pitch, DeltaTime, Speeds[4]);
    float NewYaw = FMath::FInterpTo(InternalSmoothedRot.Yaw, RawRotation.Yaw, DeltaTime, Speeds[5]);

    InternalSmoothedRot = FRotator(NewPitch, NewYaw, NewRoll);

    // 6. Output
    OutSmoothedLoc = InternalSmoothedLoc;
    OutSmoothedRot = InternalSmoothedRot;

    // 7. Update the shader material according to average sigma
    if (!DynamicMaterial && CalibrationObject)
    {
        SetupMaterial();
    }

    if (DynamicMaterial)
    {
        // Average the Sigma of X, Y, Z
        float AvgSigma = TotalSigma / 3.0f;
        DynamicMaterial->SetScalarParameterValue(FName("Sigma"), AvgSigma);
    }
}


float UAdaptiveSmoothingComponent::CalculateSigma(int32 AxisIndex, float CurrentAccel)
{
    if (!AccelerationBuffers.IsValidIndex(AxisIndex))
        return 0.0f;

    TArray<float> &Buffer = AccelerationBuffers[AxisIndex];
    // 1. Add new acceleration to buffer
    Buffer.Add(CurrentAccel);

    // 2. Manage Ring Buffer Size
    if (Buffer.Num() > WindowSize)
    {
        Buffer.RemoveAt(0);
    }

    // 3. WARM-UP LOGIC (Matching Python "min_periods = window // 2")
    // If we don't have enough history, return a value that forces "Max Speed" (Raw Tracking).
    int32 MinSamples = FMath::Max(1, WindowSize / 2); // e.g., 25 / 2 = 12 frames

    if (Buffer.Num() < MinSamples)
    {
        return 0.0f;
    }

    float Sum = 0.0f;
    for (float Val : Buffer)
        Sum += Val;
    float Mean = Sum / Buffer.Num();

    float SumSqDiff = 0.0f;
    for (float Val : Buffer)
        SumSqDiff += FMath::Square(Val - Mean);

    return FMath::Sqrt(SumSqDiff / Buffer.Num());
}

float UAdaptiveSmoothingComponent::GetSpeed_Linear(int32 AxisIndex, float Sigma)
{
    if (!LinearConfig.IsValidIndex(AxisIndex))
        return 10.0f;
    const FLinearAxisConfig &Cfg = LinearConfig[AxisIndex];
    return FMath::GetMappedRangeValueClamped(
        FVector2D(Cfg.MinSigma, Cfg.MaxSigma),
        FVector2D(Cfg.MaxSpeed, Cfg.MinSpeed),
        Sigma);
}

float UAdaptiveSmoothingComponent::GetSpeed_Sigmoid(int32 AxisIndex, float Sigma)
{
    if (!SigmoidConfig.IsValidIndex(AxisIndex))
        return 10.0f;
    const FSigmoidAxisConfig &Cfg = SigmoidConfig[AxisIndex];
    float Logistic = 1.0f / (1.0f + FMath::Exp(-Cfg.Steepness * (Sigma - Cfg.Midpoint)));
    return Cfg.MinSpeed + (Cfg.MaxSpeed - Cfg.MinSpeed) * (1.0f - Logistic);
}

float UAdaptiveSmoothingComponent::GetSpeed_Piecewise(int32 AxisIndex, float Sigma)
{
    if (!PiecewiseConfig.IsValidIndex(AxisIndex))
        return 10.0f;
    const FPiecewiseAxisConfig &Cfg = PiecewiseConfig[AxisIndex];

    if (Cfg.SigmaBreaks.Num() == 0 || Cfg.SigmaBreaks.Num() != Cfg.SpeedLevels.Num())
        return 10.0f;

    if (Sigma <= Cfg.SigmaBreaks[0])
        return Cfg.SpeedLevels[0];
    if (Sigma >= Cfg.SigmaBreaks.Last())
        return Cfg.SpeedLevels.Last();

    for (int i = 0; i < Cfg.SigmaBreaks.Num() - 1; i++)
    {
        float B0 = Cfg.SigmaBreaks[i];
        float B1 = Cfg.SigmaBreaks[i + 1];
        if (Sigma >= B0 && Sigma <= B1)
        {
            float Alpha = (Sigma - B0) / (B1 - B0);
            return FMath::Lerp(Cfg.SpeedLevels[i], Cfg.SpeedLevels[i + 1], Alpha);
        }
    }
    return Cfg.SpeedLevels.Last();
}