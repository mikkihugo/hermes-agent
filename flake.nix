{
  description = "Hermes Agent - AI agent framework by Nous Research";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    npm-lockfile-fix = {
      url = "github:jeslie0/npm-lockfile-fix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  nixConfig = {
    extra-substituters = [
      "https://cache.centralcloud.com/default"
    ];
    extra-trusted-public-keys = [
      "default:ywfU21WX06iOn2Ec2lae1jYh4w8LO4IQkmp06vJzsk8="
    ];
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      imports = [
        ./nix/packages.nix
        ./nix/overlays.nix
        ./nix/nixosModules.nix
        ./nix/checks.nix
        ./nix/devShell.nix
      ];
    };
}
